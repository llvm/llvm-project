#include <stdio.h>
#include <math.h>

#define TYPE double

template<class T>
__attribute__((noinline))
void compute_center(T _cx0, T _cy0, T _px0, T _py0, T _radius)
{

  T final_radius = (T) _radius;

  // Center of the circle
  T cx = (T) _cx0;
  T cy = (T) _cy0;

  // Distance between the center and the point for different axis
  T dx0 = (T) (_px0 - _cx0);
  T dy0 = (T) (_py0 - _cy0);

  // Euclidean distance between the center and the point
  T euclidean_dist_sqr_pc = dx0*dx0 + dy0*dy0;
  T newRadius;

  // Checking if square of Euclidean distance is greater than square of radius of circle
  if (euclidean_dist_sqr_pc > (final_radius*final_radius)) {
    // Point lies outside circle.
    T euclidean_dist_pc = sqrt((double) euclidean_dist_sqr_pc);
    newRadius = (final_radius + euclidean_dist_pc) * 0.5;

    // Computing the translation factor
    T k = (newRadius - _radius)/euclidean_dist_pc;

    // Computing the new center
    cx = _cx0 + dx0*k;
    cy = _cy0 + dy0*k;
  }

  printf("Center of the circle is (%f, %f)\n", cx, cy);
}

int main() {
  TYPE cx = 1.0;
  TYPE cy = 1.0;
  TYPE px = 4.0;
  TYPE py = 5.0;
  TYPE radius = 3.0;

  compute_center(cx, cy, px, py, radius);
  return 0;
}