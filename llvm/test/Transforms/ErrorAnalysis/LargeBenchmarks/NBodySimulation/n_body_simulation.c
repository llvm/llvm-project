#include <math.h>
#include <stdio.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define SQRT sqrt

__attribute__((noinline))
void run_simulation(TYPE x0, TYPE y0, TYPE z0, TYPE vx0, TYPE vy0, TYPE vz0) {
  // Time step
  TYPE dt = 0.1;

  // Mass of the Sun in units. The sun is assumed to be the origin.
  TYPE solarMass = 39.47841760435743;

  // Co-ordinates of the Planet with an assumed mass of 1.0 unit
  TYPE x = x0, y = y0, z = z0;

  // Velocity of the Planet
  TYPE vx = vx0, vy = vy0, vz = vz0;

  // Run the simulation for 100 iterations
  for (int i = 0; i < 100; i++) {
    // Computing the Euclidean distance between the Planet and the Sun
    TYPE distance_cubed = SQRT((((x * x) + (y * y)) + (z * z)));

    // Computing the acceleration
    TYPE mag = dt / distance_cubed;

    // Computing new velocities
    TYPE vx_New = vx - ((x * solarMass) * mag);
    TYPE vy_New = vy - ((y * solarMass) * mag);
    TYPE vz_New = vz - ((z * solarMass) * mag);

    // Computing new co-ordinates of the Planet
    TYPE x_New = x + (dt * vx_New);
    TYPE y_New = y + (dt * vy_New);
    TYPE z_New = z + (dt * vz_New);

    // Updating the velocities
    vx = vx - ((x * solarMass) * mag);
    vy = vy - ((y * solarMass) * mag);
    vz = vz - ((z * solarMass) * mag);

    // Updating the co-ordinates
    x = x_New;
    y = y_New;
    z = z_New;
  }

  printf("x = "PRINT_PRECISION_FORMAT"\n", x);
  printf("y = "PRINT_PRECISION_FORMAT"\n", y);
  printf("z = "PRINT_PRECISION_FORMAT"\n", z);
}

int main() {
  TYPE x0 = 1.0;
  TYPE y0 = 1.0;
  TYPE z0 = 1.0;
  TYPE vx0 = 1.0;
  TYPE vy0 = 1.0;
  TYPE vz0 = 1.0;
  run_simulation(x0, y0, z0, vx0, vy0, vz0);

  return 0;
}