// RUN: %libomptarget-compile-run-and-check-generic
// Tests non-contiguous array sections with complex expression-based count
// scenarios including multiple struct arrays and non-zero offset.

#include <omp.h>
#include <stdio.h>

struct Data {
  int offset;
  int len;
  double arr[20];
};

void test_1_complex_count_expressions() {
  struct Data s1, s2;
  s1.len = 10;
  s2.len = 10;

  // Initialize on device
#pragma omp target map(tofrom : s1, s2)
  {
    for (int i = 0; i < s1.len; i++) {
      s1.arr[i] = i;
    }
    for (int i = 0; i < s2.len; i++) {
      s2.arr[i] = i * 10;
    }
  }

  // Test FROM: Update multiple struct arrays with complex count expressions
#pragma omp target data map(to : s1, s2)
  {
#pragma omp target
    {
      for (int i = 0; i < s1.len; i++) {
        s1.arr[i] += i;
      }
      for (int i = 0; i < s2.len; i++) {
        s2.arr[i] += i * 10;
      }
    }

    // Complex count: (len-2)/2 and len*2/5
#pragma omp target update from(s1.arr[0 : (s1.len - 2) / 2 : 2],               \
                                   s2.arr[0 : s2.len * 2 / 5 : 2])
  }

  printf("Test 1 - complex count expressions (from):\n");
  printf("s1 results:\n");
  for (int i = 0; i < s1.len; i++)
    printf("%f\n", s1.arr[i]);

  printf("s2 results:\n");
  for (int i = 0; i < s2.len; i++)
    printf("%f\n", s2.arr[i]);

  // Reset for TO test - initialize on host
  for (int i = 0; i < s1.len; i++) {
    s1.arr[i] = i * 2;
  }
  for (int i = 0; i < s2.len; i++) {
    s2.arr[i] = i * 20;
  }

  // Modify host data
  for (int i = 0; i < (s1.len - 2) / 2; i++) {
    s1.arr[i * 2] = i + 100;
  }
  for (int i = 0; i < s2.len * 2 / 5; i++) {
    s2.arr[i * 2] = i + 50;
  }

  // Test TO: Update with complex count expressions
#pragma omp target data map(to : s1, s2)
  {
#pragma omp target update to(s1.arr[0 : (s1.len - 2) / 2 : 2],                 \
                                 s2.arr[0 : s2.len * 2 / 5 : 2])

#pragma omp target
    {
      for (int i = 0; i < s1.len; i++) {
        s1.arr[i] += 100;
      }
      for (int i = 0; i < s2.len; i++) {
        s2.arr[i] += 100;
      }
    }
  }

  printf("Test 1 - complex count expressions (to):\n");
  printf("s1 results:\n");
  for (int i = 0; i < s1.len; i++)
    printf("%f\n", s1.arr[i]);

  printf("s2 results:\n");
  for (int i = 0; i < s2.len; i++)
    printf("%f\n", s2.arr[i]);
}

void test_2_complex_count_with_offset() {
  struct Data s1, s2;
  s1.offset = 2;
  s1.len = 10;
  s2.offset = 1;
  s2.len = 10;

  // Initialize on device
#pragma omp target map(tofrom : s1, s2)
  {
    for (int i = 0; i < s1.len; i++) {
      s1.arr[i] = i;
    }
    for (int i = 0; i < s2.len; i++) {
      s2.arr[i] = i * 10;
    }
  }

  // Test FROM: Complex count with offset
#pragma omp target data map(to : s1, s2)
  {
#pragma omp target
    {
      for (int i = 0; i < s1.len; i++) {
        s1.arr[i] += i;
      }
      for (int i = 0; i < s2.len; i++) {
        s2.arr[i] += i * 10;
      }
    }

    // Count: (len-offset)/2 with stride 2
#pragma omp target update from(                                                \
        s1.arr[s1.offset : (s1.len - s1.offset) / 2 : 2],                      \
            s2.arr[s2.offset : (s2.len - s2.offset) / 2 : 2])
  }

  printf("Test 2 - complex count with offset (from):\n");
  printf("s1 results:\n");
  for (int i = 0; i < s1.len; i++)
    printf("%f\n", s1.arr[i]);

  printf("s2 results:\n");
  for (int i = 0; i < s2.len; i++)
    printf("%f\n", s2.arr[i]);

  // Reset for TO test - initialize on host
  for (int i = 0; i < s1.len; i++) {
    s1.arr[i] = i * 2;
  }
  for (int i = 0; i < s2.len; i++) {
    s2.arr[i] = i * 20;
  }

  // Modify host data
  for (int i = 0; i < (s1.len - s1.offset) / 2; i++) {
    s1.arr[s1.offset + i * 2] = i + 100;
  }
  for (int i = 0; i < (s2.len - s2.offset) / 2; i++) {
    s2.arr[s2.offset + i * 2] = i + 50;
  }

  // Test TO: Update with complex count and offset
#pragma omp target data map(to : s1, s2)
  {
#pragma omp target update to(                                                  \
        s1.arr[s1.offset : (s1.len - s1.offset) / 2 : 2],                      \
            s2.arr[s2.offset : (s2.len - s2.offset) / 2 : 2])

#pragma omp target
    {
      for (int i = 0; i < s1.len; i++) {
        s1.arr[i] += 100;
      }
      for (int i = 0; i < s2.len; i++) {
        s2.arr[i] += 100;
      }
    }
  }

  printf("Test 2 - complex count with offset (to):\n");
  printf("s1 results:\n");
  for (int i = 0; i < s1.len; i++)
    printf("%f\n", s1.arr[i]);

  printf("s2 results:\n");
  for (int i = 0; i < s2.len; i++)
    printf("%f\n", s2.arr[i]);
}

// CHECK: Test 1 - complex count expressions (from):
// CHECK: s1 results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 9.000000
// CHECK: s2 results:
// CHECK-NEXT: 0.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 40.000000
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 70.000000
// CHECK-NEXT: 80.000000
// CHECK-NEXT: 90.000000
// CHECK: Test 1 - complex count expressions (to):
// CHECK: s1 results:
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 101.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 102.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 103.000000
// CHECK-NEXT: 14.000000
// CHECK-NEXT: 16.000000
// CHECK-NEXT: 18.000000
// CHECK: s2 results:
// CHECK-NEXT: 50.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 51.000000
// CHECK-NEXT: 60.000000
// CHECK-NEXT: 52.000000
// CHECK-NEXT: 100.000000
// CHECK-NEXT: 53.000000
// CHECK-NEXT: 140.000000
// CHECK-NEXT: 160.000000
// CHECK-NEXT: 180.000000
// CHECK: Test 2 - complex count with offset (from):
// CHECK: s1 results:
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
// CHECK: s2 results:
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
// CHECK: Test 2 - complex count with offset (to):
// CHECK: s1 results:
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
// CHECK: s2 results:
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

int main() {
  test_1_complex_count_expressions();
  test_2_complex_count_with_offset();
  return 0;
}
