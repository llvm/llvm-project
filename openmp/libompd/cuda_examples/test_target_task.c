#include <stdio.h>
#include <stdint.h>

#pragma omp declare target
void task1() {
  printf("Hello from Task 1\n");
  uint32_t enter_frame = 0;
  for(;1;) {
  }
}
void task2() {
  printf("Hello from Task 2\n");
  for(;1;) {
  }
}
#pragma omp end declare target

int main() {
  #pragma omp target
  {
  #pragma omp parallel num_threads(4)
  {
    #pragma omp single
    {
      #pragma omp task
      task1();
      #pragma omp task
      task2();
    }
  }
  }
  return 0;
}
