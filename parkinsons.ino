#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

int vibrationMotorPin1 = 7;  // Motor 1 connected to Pin 7
int vibrationMotorPin2 = 6;  // Motor 2 connected to Pin 6
int vibrationMotorPin3 = 5;  // Motor 3 connected to Pin 5
int vibrationMotorPin4 = 4;  // Motor 4 connected to Pin 4
int redPin = 13;             // Red LED connected to Pin 13
int greenPin = 12;           // Green LED connected to Pin 12
int bluePin = 11;            // Blue LED connected to Pin 11
int buzzerPin = 2;           // Buzzer connected to Pin 2

int signal; // Variable to store the signal from the computer

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  // Initialize MPU6050
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1); // Halt the program if initialization fails
  }

  pinMode(vibrationMotorPin1, OUTPUT);
  pinMode(vibrationMotorPin2, OUTPUT);
  pinMode(vibrationMotorPin3, OUTPUT);
  pinMode(vibrationMotorPin4, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  pinMode(buzzerPin, OUTPUT); // Set buzzer pin as output
}

void loop() {
  // Check for signals from the computer
  if (Serial.available() > 0) {
    char incomingChar = Serial.read(); // Read the incoming byte
    if (incomingChar >= '0' && incomingChar <= '9') {
      signal = incomingChar - '0'; // Convert char to integer
    }
  }

  // Variables to store accelerometer and gyroscope data
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  
  // Get accelerometer and gyroscope data
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Calculate tilt angle (using y-axis as an example)
  float tiltAngle = atan2(ay, az) * 180 / PI;

  // Set color based on tilt and signal values
  bool isHighTilt = false;
  if (abs(tiltAngle) < 10) {
    setColor(0, 255, 0); // Green for little/no tilt
  } else if (abs(tiltAngle) >= 10 && abs(tiltAngle) < 20) {
    setColor(0, 0, 255); // Blue for medium tilt
  } else {
    setColor(255, 0, 0); // Red for high tilt
    isHighTilt = true; // Flag for high tilt
  }

  // Override color based on signal only if tilt is minimal
  if (abs(tiltAngle) < 10) {
    if (signal == 0) {
      setColor(0, 255, 0); // Green for signal 0
    } else if (signal == 1 || signal == 2) {
      setColor(0, 0, 255); // Blue for signal 1 or 2
    } else if (signal >= 3) {
      setColor(255, 0, 0); // Red for signal > 3
    }
  }

  // Adjust tilt threshold, vibration intensity, and buzzer behavior based on severity
  int tiltThreshold;
  int vibrationIntensity1, vibrationIntensity2, vibrationIntensity3, vibrationIntensity4;

  if (signal == 0) { // No symptoms detected
    tiltThreshold = 15;
    vibrationIntensity1 = 100; // Low intensity
    vibrationIntensity2 = 100; // Low intensity
    vibrationIntensity3 = 100; // Low intensity
    vibrationIntensity4 = 100; // Low intensity
  } else if (signal == 1 || signal == 2) { // Mild symptoms
    tiltThreshold = 12;
    vibrationIntensity1 = 150; // Medium intensity
    vibrationIntensity2 = 150; // Medium intensity
    vibrationIntensity3 = 150; // Medium intensity
    vibrationIntensity4 = 150; // Medium intensity
  } else if (signal >= 3) { // Moderate to severe symptoms
    tiltThreshold = 10;
    vibrationIntensity1 = 255; // High intensity
    vibrationIntensity2 = 255; // High intensity
    vibrationIntensity3 = 255; // High intensity
    vibrationIntensity4 = 255; // High intensity
  }

  // Check if tilt exceeds the adjusted threshold OR signal indicates a symptom level
  if (abs(tiltAngle) > tiltThreshold || signal >= 1) {  // Adjusted condition
    analogWrite(vibrationMotorPin1, vibrationIntensity1); // Activate motor 1
    analogWrite(vibrationMotorPin2, vibrationIntensity2); // Activate motor 2
    analogWrite(vibrationMotorPin3, vibrationIntensity3); // Activate motor 3
    analogWrite(vibrationMotorPin4, vibrationIntensity4); // Activate motor 4
  } else {
    // Deactivate all motors if tilt is below threshold
    digitalWrite(vibrationMotorPin1, LOW);
    digitalWrite(vibrationMotorPin2, LOW);
    digitalWrite(vibrationMotorPin3, LOW);
    digitalWrite(vibrationMotorPin4, LOW);
  }

  // Activate buzzer only if signal > 3 or high tilt (red LED)
  if (signal >= 3 || isHighTilt) {
    digitalWrite(buzzerPin, HIGH); // Activate buzzer
  } else {
    digitalWrite(buzzerPin, LOW);  // Turn off buzzer
  }

  delay(100); // Sampling delay
}

// Function to set the RGB LED color
void setColor(int red, int green, int blue) {
  analogWrite(redPin, red);
  analogWrite(greenPin, green);
  analogWrite(bluePin, blue);
}
