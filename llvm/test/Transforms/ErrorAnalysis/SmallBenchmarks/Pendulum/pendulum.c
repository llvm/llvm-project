#include <stdio.h>
#include <math.h>
#include <time.h>

#define TYPE float
#define PRINT_PRECISION_FORMAT "%0.7f"
#define SIN sinf

// Inputs
#define INITIAL_ANGLE 2.0
#define INITIAL_ANGULAR_VELOCITY 0.5
#define N 2500

#define EULER_STEP_SIZE 0.05


__attribute__((noinline))
double eulers_pendulum(double t, double w) {
  // Step size of Euler's method
  double h = EULER_STEP_SIZE;

  // Pendulum parameters
  double L = 2.0; // Length of pendulum
  double m = 1.5; // Mass of pendulum
  double g = 9.80665; // Acceleration due to gravity

  for (int I = 0; I < N; I++) {
    // Derivative of the angular velocity
    double dw = (-g / L) * SIN(t);

    // Calculate the new angle and angular velocity
    t = t + (h*w);
    w = w + (h*dw);
//    printf("Angle: "PRINT_PRECISION_FORMAT"\n", t);
  }

  return t;
}

int main() {
  // Calculate Time
  clock_t t;
  t = clock();

  double FinalAngle = eulers_pendulum(INITIAL_ANGLE, INITIAL_ANGULAR_VELOCITY);
  t = clock() - t;
  printf("Angle:"PRINT_PRECISION_FORMAT"\n", FinalAngle);
  printf("Time: %f\n", ((double)t)/CLOCKS_PER_SEC);

  return 0;
}