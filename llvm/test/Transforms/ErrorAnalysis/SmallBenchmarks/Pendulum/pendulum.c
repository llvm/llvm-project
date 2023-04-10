#include <stdio.h>
#include <math.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"

#define EULER_STEP_SIZE 0.05

__attribute__((noinline))
double eulers_pendulum(double t, double w, int N) {
  // Step size of Euler's method
  double h = EULER_STEP_SIZE;

  // Pendulum parameters
  double L = 2.0; // Length of pendulum
  double m = 1.5; // Mass of pendulum
  double g = 9.80665; // Acceleration due to gravity

  for (int I = 0; I < N; I++) {
    // Derivative of the angular velocity
    double dw = (-g / L) * sin(t);

    // Calculate the new angle and angular velocity
    t = t + (h*w);
    w = w + (h*dw);
//    printf("Angle: "PRINT_PRECISION_FORMAT"\n", t);
  }

  return t;
}

int main() {
  double InitialAngle = 2.0;
  double InitialAngularVelocity = 0.5;
  double NumberOfIterations = 100.0;

  double FinalAngle = eulers_pendulum(InitialAngle, InitialAngularVelocity, NumberOfIterations);
  printf("Angle:"PRINT_PRECISION_FORMAT"\n", FinalAngle);

  return 0;
}