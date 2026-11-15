// RUN: %libomptarget-compile-run-and-check-generic
// UNSUPPORTED: apu

#include <assert.h>
#include <stdio.h>

static void print(int darr[2][4]) {
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++)
      printf("\t%d", darr[i][j]);
    printf("\n");
  }
}

static void strided_inner_dimension(void) {
  // A strided inner dimension must not be merged with the outer one, even
  // though count * stride of the inner dimension equals the outer stride.
  int darr[2][4] = {0};

#pragma omp target enter data map(to : darr)
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      darr[i][j] = 10 * i + j;

#pragma omp target update to(darr[0 : 2][1 : 2 : 2])
#pragma omp target exit data map(from : darr)
  printf("Only slice (to)\n");
  print(darr);

#pragma omp target enter data map(alloc : darr)
#pragma omp target
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      darr[i][j] = 100 + 10 * i + j;

#pragma omp target update from(darr[0 : 2][1 : 2 : 2])
#pragma omp target exit data map(delete : darr)
  printf("Only slice (from)\n");
  print(darr);
}

static void gap_between_contiguous_dimensions(void) {
  // A contiguous pair of outer dimensions must not be merged across a gap
  // between inner dimensions. This is the C equivalent of the Fortran section
  // arr(:,3:5,:,:) for integer(4) :: arr(4,5,3,2).
  int data[2][3][5][4];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 5; ++k)
        for (int l = 0; l < 4; ++l)
          data[i][j][k][l] = -1;

#pragma omp target data map(to : data)
  {
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 5; ++k)
          for (int l = 0; l < 4; ++l)
            data[i][j][k][l] = 7;

#pragma omp target update to(data[0 : 2][0 : 3][2 : 3][0 : 4])
#pragma omp target update from(data)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 5; ++k)
          for (int l = 0; l < 4; ++l)
            assert(data[i][j][k][l] == (k >= 2 ? 7 : -1));

    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 5; ++k)
          for (int l = 0; l < 4; ++l)
            data[i][j][k][l] = 1000 * i + 100 * j + 10 * k + l;
#pragma omp target update to(data)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 5; ++k)
          for (int l = 0; l < 4; ++l)
            data[i][j][k][l] = -9;

#pragma omp target update from(data[0 : 2][0 : 3][2 : 3][0 : 4])
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 5; ++k)
          for (int l = 0; l < 4; ++l)
            assert(data[i][j][k][l] ==
                   (k >= 2 ? 1000 * i + 100 * j + 10 * k + l : -9));
  }
  puts("Four-dimensional slice: PASS");
}

int main(void) {
  strided_inner_dimension();
  gap_between_contiguous_dimensions();
  return 0;
}

//      CHECK: Only slice (to)
// CHECK-NEXT:         0       1       0       3
// CHECK-NEXT:         0       11      0       13
// CHECK-NEXT: Only slice (from)
// CHECK-NEXT:         0       101     0       103
// CHECK-NEXT:         0       111     0       113
// CHECK-NEXT: Four-dimensional slice: PASS
