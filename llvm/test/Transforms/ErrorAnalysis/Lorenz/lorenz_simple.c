//
// Created by tanmay on 8/10/22.
//

#include <stdio.h>
#include <stdlib.h>

int main() {
  float x_1 = 0, y_1 = 0, z_1 = 0;
  float x_0 = 1.2;
  float y_0 = 1.3;
  float z_0 = 1.6;

  for(int i = 0; i < 3; i++) {
    x_1 = x_0 + 10*(y_0-x_0)*0.005;
    y_1 = y_0 + (100*x_0 - y_0 - x_0*z_0)*0.005;
    z_1 = z_0 + (x_0*y_0 - 2.666667*z_0)*0.005;

    printf("x_0 = %0.7f\n", x_0);
    printf("y_0 = %0.7f\n", y_0);
    printf("z_0 = %0.7f\n\n", z_0);

    x_0 = x_1 + 10*(y_1-x_1)*0.005;
    y_0 = y_1 + (28*x_1 - y_1 - x_1*z_1)*0.005;
    z_0 = z_1 + (x_1*y_1 - 2.666667*z_1)*0.005;
    fAFfp32markForResult(x_0);
    fAFfp32markForResult(y_0);
    fAFfp32markForResult(z_0);

    printf("x_1 = %0.7f\n", x_1);
    printf("y_1 = %0.7f\n", y_1);
    printf("z_1 = %0.7f\n\n", z_1);
  }
}