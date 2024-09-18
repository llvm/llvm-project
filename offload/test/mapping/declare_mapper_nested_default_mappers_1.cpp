// RUN: %libomptarget-compilexx-run-and-check-generic

// UNSUPPORTED: amdgcn-amd-amdhsa

extern "C" int printf(const char *, ...);

typedef struct {
  int a;
} C;
#pragma omp declare mapper(C s) map(to : s.a)

typedef struct {
  int e;
  C f;
  int h;
} D;

int main() {
  D sa[10];
  sa[1].e = 111;
  sa[1].f.a = 222;

  // CHECK: 111 222
  printf("%d %d \n", sa[1].e, sa[1].f.a);
#pragma omp target map(tofrom : sa[0 : 2])
  {
    // CHECK: 111
    printf("%d \n", sa[1].e);
    sa[0].e = 333;
    sa[1].f.a = 444;
    // CHECK: 333 444
    printf("%d %d \n", sa[0].e, sa[1].f.a);
  }
  // CHECK: 333 222
  printf("%d %d \n", sa[0].e, sa[1].f.a);
}
