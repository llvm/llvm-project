// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// This test ensures that when mapping `s.p` and `s.p[0:10] via a mapper, they
// occupy different storage blocks in memory, so deleting `s.p` does not delete
// `s.p[0:10]`.

int x[10];

typedef struct {
  int *p;
} S;

#pragma omp declare mapper(default : S s) map(s.p, s.p[0 : 10])

S s1;

void print_status(void *p, const char *name) {
  int present = omp_target_is_present(p, omp_get_default_device());
  printf("%s is %spresent\n", name, present ? "" : "not ");
}
int main() {
  s1.p = (int *)&x;

#pragma omp target enter data map(alloc : s1)
  printf("After mapping ptr and ptee\n");
  print_status(&s1.p, "p");       // CHECK: p is present
  print_status(&s1.p[0], "p[0]"); // CHECK: p[0] is present
  printf("\n");

#pragma omp target exit data map(delete : s1.p)
  printf("After deleting ptr\n");
  print_status(&s1.p, "p");       // CHECK: p is not present
  print_status(&s1.p[0], "p[0]"); // CHECK: p[0] is present
  printf("\n");

#pragma omp target exit data map(delete : s1.p[0 : 10])
  printf("After deleting ptee\n");
  print_status(&s1.p, "p");       // CHECK: p is not present
  print_status(&s1.p[0], "p[0]"); // CHECK: p[0] is not present
}
