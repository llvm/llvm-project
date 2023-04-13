#include <math.h>
#include <stdio.h>

#define TYPE float
#define PRINT_PRECISION_FORMAT "%0.7f"
#define SQRT sqrtf

// Inputs
#define n 650

// Co-ordinates of the Planet with an assumed mass of 1.0 unit
#define x 0.637385368347168
#define y 0.432825654745102
#define z 0.721410274505615
// Velocity of the Planet
#define vx -1.837728261947632
#define vy -1.382405281066895
#define vz -2.091578245162964

__attribute__((noinline))
void run_simulation(TYPE x0, TYPE y0, TYPE z0, TYPE vx0, TYPE vy0, TYPE vz0) {
  // Time step
  TYPE dt = 0.1;

  // Mass of the Sun in units. The sun is assumed to be the origin.
  TYPE solarMass = 39.47841760435743;

  // Run the simulation for 100 iterations
  for (int i = 0; i < n; i++) {
    // Computing the Euclidean distance between the Planet and the Sun
    TYPE distance_cubed = SQRT((((x0 * x0) + (y0 * y0)) + (z0 * z0)));

    // Computing the acceleration
    TYPE mag = dt / distance_cubed;

    // Computing new velocities
    TYPE vx_New = vx0 - ((x0 * solarMass) * mag);
    TYPE vy_New = vy0 - ((y0 * solarMass) * mag);
    TYPE vz_New = vz0 - ((z0 * solarMass) * mag);

    // Computing new co-ordinates of the Planet
    TYPE x_New = x0 + (dt * vx_New);
    TYPE y_New = y0 + (dt * vy_New);
    TYPE z_New = z0 + (dt * vz_New);

    // Updating the velocities
    vx0 -= ((x0 * solarMass) * mag);
    vy0 -= ((y0 * solarMass) * mag);
    vz0 -= ((z0 * solarMass) * mag);

    // Updating the co-ordinates
    x0 = x_New;
    y0 = y_New;
    z0 = z_New;
  }

//  printf("x = "PRINT_PRECISION_FORMAT"\n", x);
//  printf("y = "PRINT_PRECISION_FORMAT"\n", y);
//  printf("z = "PRINT_PRECISION_FORMAT"\n", z);
}

int main() {
  run_simulation(x, y, z, vx, vy, vz);

  return 0;
}