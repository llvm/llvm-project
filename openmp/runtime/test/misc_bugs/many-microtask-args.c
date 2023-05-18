// RUN: %libomp-compile-and-run
#include <stdio.h>

int main()
{

  int i;
  int i1 = 0;
  int i2 = 1;
  int i3 = 2;
  int i4 = 3;
  int i5 = 4;
  int i6 = 6;
  int i7 = 7;
  int i8 = 8;
  int i9 = 9;
  int i10 = 10;
  int i11 = 11;
  int i12 = 12;
  int i13 = 13;
  int i14 = 14;
  int i15 = 15;
  int i16 = 16;
 
  int r = 0; 
  #pragma omp parallel for firstprivate(i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16) reduction(+:r)
  for (i = 0; i < i16; i++) {
    r += i + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + i10 + i11 + i12 + i13 + i14 + i15 + i16;
  }

  int rf = 2216;
  if (r != rf) {
    fprintf(stderr, "r should be %d but instead equals %d\n", rf, r);
    return 1;
  }

  return 0;
}

