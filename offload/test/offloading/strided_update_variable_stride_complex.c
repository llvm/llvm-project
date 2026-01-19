// RUN: %libomptarget-compile-run-and-check-generic
// Tests complex variable stride patterns with multiple arrays and offsets.

#include <omp.h>
#include <stdio.h>

struct Data {
  int offset;
  int stride;
  double arr[20];
};

int main() {
  struct Data d1, d2;
  int len1 = 10;
  int len2 = 10;

  // Test 1: Complex stride expressions
  int base_stride = 1;
  int multiplier = 2;
  d1.stride = 2;
  d2.stride = 3;

  // Initialize on device
#pragma omp target map(tofrom : d1, d2, base_stride, multiplier)
  {
    for (int i = 0; i < len1; i++) {
      d1.arr[i] = i * 3;
    }
    for (int i = 0; i < len2; i++) {
      d2.arr[i] = i * 30;
    }
  }

  // Test FROM: Complex stride expressions
#pragma omp target data map(to : d1, d2, base_stride, multiplier)
  {
#pragma omp target
    {
      for (int i = 0; i < len1; i++) {
        d1.arr[i] += i * 3;
      }
      for (int i = 0; i < len2; i++) {
        d2.arr[i] += i * 30;
      }
    }

    // Stride expressions: base_stride*multiplier and (d2.stride+1)/2
#pragma omp target update from(d1.arr[0 : 5 : base_stride * multiplier],       \
                                   d2.arr[0 : 3 : (d2.stride + 1) / 2])
  }

  printf("Test 1 - complex stride expressions (from):\n");
  printf("d1 results (stride=%d*%d=%d):\n", base_stride, multiplier,
         base_stride * multiplier);
  for (int i = 0; i < len1; i++)
    printf("%f\n", d1.arr[i]);

  printf("d2 results (stride=(%d+1)/2=%d):\n", d2.stride, (d2.stride + 1) / 2);
  for (int i = 0; i < len2; i++)
    printf("%f\n", d2.arr[i]);

  // Reset for TO test
#pragma omp target map(tofrom : d1, d2)
  {
    for (int i = 0; i < len1; i++) {
      d1.arr[i] = i * 4;
    }
    for (int i = 0; i < len2; i++) {
      d2.arr[i] = i * 40;
    }
  }

  // Modify host data with stride expressions
  int stride1 = base_stride * multiplier;
  int stride2 = (d2.stride + 1) / 2;
  for (int i = 0; i < 5; i++) {
    d1.arr[i * stride1] = i + 200;
  }
  for (int i = 0; i < 3; i++) {
    d2.arr[i * stride2] = i + 150;
  }

  // Test TO: Update with complex stride expressions
#pragma omp target data map(to : d1, d2, base_stride, multiplier)
  {
#pragma omp target update to(d1.arr[0 : 5 : base_stride * multiplier],         \
                                 d2.arr[0 : 3 : (d2.stride + 1) / 2])

#pragma omp target
    {
      for (int i = 0; i < len1; i++) {
        d1.arr[i] += 200;
      }
      for (int i = 0; i < len2; i++) {
        d2.arr[i] += 200;
      }
    }
  }

  printf("Test 1 - complex stride expressions (to):\n");
  printf("d1 results (stride=%d*%d=%d):\n", base_stride, multiplier,
         base_stride * multiplier);
  for (int i = 0; i < len1; i++)
    printf("%f\n", d1.arr[i]);

  printf("d2 results (stride=(%d+1)/2=%d):\n", d2.stride, (d2.stride + 1) / 2);
  for (int i = 0; i < len2; i++)
    printf("%f\n", d2.arr[i]);

  // Test 2: Variable stride with non-zero offset
  d1.offset = 2;
  d1.stride = 2;
  d2.offset = 1;
  d2.stride = 2;

  // Initialize on device
#pragma omp target map(tofrom : d1, d2, len1, len2)
  {
    for (int i = 0; i < len1; i++) {
      d1.arr[i] = i;
    }
    for (int i = 0; i < len2; i++) {
      d2.arr[i] = i * 10;
    }
  }

  // Test FROM: Variable stride with offset
#pragma omp target data map(to : d1, d2, len1, len2)
  {
#pragma omp target
    {
      for (int i = 0; i < len1; i++) {
        d1.arr[i] += i;
      }
      for (int i = 0; i < len2; i++) {
        d2.arr[i] += i * 10;
      }
    }

#pragma omp target update from(d1.arr[d1.offset : 4 : d1.stride],              \
                                   d2.arr[d2.offset : 4 : d2.stride])
  }

  printf("Test 2 - variable stride with offset (from):\n");
  printf("d1 results:\n");
  for (int i = 0; i < len1; i++)
    printf("%f\n", d1.arr[i]);

  printf("d2 results:\n");
  for (int i = 0; i < len2; i++)
    printf("%f\n", d2.arr[i]);

  // Reset for TO test
#pragma omp target map(tofrom : d1, d2)
  {
    for (int i = 0; i < len1; i++) {
      d1.arr[i] = i * 2;
    }
    for (int i = 0; i < len2; i++) {
      d2.arr[i] = i * 20;
    }
  }

  // Modify host data
  for (int i = 0; i < 4; i++) {
    d1.arr[d1.offset + i * d1.stride] = i + 100;
  }
  for (int i = 0; i < 4; i++) {
    d2.arr[d2.offset + i * d2.stride] = i + 50;
  }

  // Test TO: Update with variable stride and offset
#pragma omp target data map(to : d1, d2)
  {
#pragma omp target update to(d1.arr[d1.offset : 4 : d1.stride],                \
                                 d2.arr[d2.offset : 4 : d2.stride])

#pragma omp target
    {
      for (int i = 0; i < len1; i++) {
        d1.arr[i] += 100;
      }
      for (int i = 0; i < len2; i++) {
        d2.arr[i] += 100;
      }
    }
  }

  printf("Test 2 - variable stride with offset (to):\n");
  printf("d1 results:\n");
  for (int i = 0; i < len1; i++)
    printf("%f\n", d1.arr[i]);

  printf("d2 results:\n");
  for (int i = 0; i < len2; i++)
    printf("%f\n", d2.arr[i]);

  return 0;
}

// CHECK: Test 1 - complex stride expressions (from):
// CHECK: d1 results (stride=1*2=2):
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 18.000000
// CHECK-NEXT: 24.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 18.000000
// CHECK-NEXT: 21.000000
// CHECK-NEXT: 24.000000
// CHECK-NEXT: 27.000000
// CHECK: d2 results (stride=(3+1)/2=2):
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 120.000000
// CHECK-NEXT: 90.000000
// CHECK-NEXT: 120.000000
// CHECK-NEXT: 150.000000
// CHECK-NEXT: 180.000000
// CHECK-NEXT: 210.000000
// CHECK-NEXT: 240.000000
// CHECK-NEXT: 270.000000
// CHECK: Test 1 - complex stride expressions (to):
// CHECK: d1 results (stride=1*2=2):
// CHECK-NEXT: 200.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 201.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 202.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 203.000000
// CHECK-NEXT: 28.000000
// CHECK-NEXT: 204.000000
// CHECK-NEXT: 36.000000
// CHECK: d2 results (stride=(3+1)/2=2):
// CHECK-NEXT: 150.000000
// CHECK-NEXT: 40.000000
// CHECK-NEXT: 151.000000
// CHECK-NEXT: 120.000000
// CHECK-NEXT: 152.000000
// CHECK-NEXT: 200.000000
// CHECK-NEXT: 240.000000
// CHECK-NEXT: 280.000000
// CHECK-NEXT: 320.000000
// CHECK-NEXT: 360.000000
// CHECK: Test 2 - variable stride with offset (from):
// CHECK: d1 results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 1.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 14.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 18.000000
// CHECK: d2 results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 40.000000
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 140.000000
// CHECK-NEXT: 80.000000
// CHECK-NEXT: 90.000000
// CHECK: Test 2 - variable stride with offset (to):
// CHECK: d1 results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 101.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 14.000000
// CHECK-NEXT: 103.000000
// CHECK-NEXT: 18.000000
// CHECK: d2 results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 50.000000
// CHECK-NEXT: 40.000000
// CHECK-NEXT: 51.000000
// CHECK-NEXT: 80.000000
// CHECK-NEXT: 52.000000
// CHECK-NEXT: 120.000000
// CHECK-NEXT: 53.000000
// CHECK-NEXT: 160.000000
// CHECK-NEXT: 180.000000
