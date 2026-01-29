// RUN: %libomptarget-compilexx-run-and-check-generic
// XFAIL: intelgpu

#include <omp.h>
#include <stdio.h>

#define TRUE 1
#define FALSE 0

struct TY1 {
  int i1, i2, i3;
  static constexpr auto name = "TY1";
};
struct TY2 {
  int i1, i2, i3;
  static constexpr auto name = "TY2";
};

// TY1 is not mapped, TY2 is
#pragma omp declare mapper(TY2 t) map(to : t.i1) map(from : t.i3)

struct TY3 {
  TY2 n;
  static constexpr auto name = "TY3";
};
struct TY4 {
  int a;
  TY2 n;
  int b;
  static constexpr auto name = "TY4";
};

template <typename T> int testType() {
  T t1[2], t2[3], t3[4];
  for (int i = 0; i < 2; i++)
    t1[i].i1 = t3[i].i1 = 1;

#pragma omp target map(tofrom : t1, t2, t3)
  for (int i = 0; i < 2; i++) {
    t1[i].i3 = t3[i].i3 = t1[i].i1;
    t1[i].i1 = t3[i].i1 = 7;
  }

  for (int i = 0; i < 2; i++) {
    if (t1[i].i3 != 1) {
      printf("failed %s. t1[%d].i3 (%d) != t1[%d].i1 (%d)\n", T::name, i,
             t1[i].i3, i, t1[i].i1);
      return 1;
    }
    if (t3[i].i3 != 1) {
      printf("failed %s. t3[%d].i3 (%d) != t3[%d].i1 (%d)\n", T::name, i,
             t3[i].i3, i, t3[i].i1);
      return 1;
    }
  }

  int pt0 = omp_target_is_present(&t1[0], omp_get_default_device());
  int pt1 = omp_target_is_present(&t2[1], omp_get_default_device());
  int pt2 = omp_target_is_present(&t3[2], omp_get_default_device());

  printf("present check for %s: t1 %i, t2 %i, t3 %i, expected 3x 0\n", T::name,
         pt0, pt1, pt2);
  return pt0 + pt1 + pt2;
}

template <typename T> int testTypeNestedPtr(T t1[2], T t2[3], T t3[4]) {
  for (int i = 0; i < 2; i++)
    t1[i].n.i1 = t3[i].n.i1 = 1;

#pragma omp target map(tofrom : t1[0 : 2], t2[0 : 3], t3[0 : 4])
  for (int i = 0; i < 2; i++) {
    t1[i].n.i3 = t3[i].n.i3 = t1[i].n.i1;
    t1[i].n.i1 = t3[i].n.i1 = 7;
  }

  for (int i = 0; i < 2; i++) {
    if (t1[i].n.i3 != t1[i].n.i1) {
      printf("failed %s-ptr. t1[%d].i3 (%d) != t1[%d].i1 (%d)\n", T::name, i,
             t1[i].n.i3, i, t1[i].n.i1);
      return 1;
    }
    if (t3[i].n.i3 != t3[i].n.i1) {
      printf("failed %s-ptr. t3[%d].i3 (%d) != t3[%d].i1 (%d)\n", T::name, i,
             t3[i].n.i3, i, t3[i].n.i1);
      return 1;
    }
  }

  int pt0 = omp_target_is_present(&t1[0], omp_get_default_device());
  int pt1 = omp_target_is_present(&t2[1], omp_get_default_device());
  int pt2 = omp_target_is_present(&t3[2], omp_get_default_device());

  printf("present check for %s-ptr: t1 %i, t2 %i, t3 %i, expected 3x 0\n",
         T::name, pt0, pt1, pt2);
  return pt0 + pt1 + pt2;
}

template <typename T> int testTypeNested() {
  T t1[2], t2[3], t3[4];
  testTypeNestedPtr(t1, t2, t3);
  for (int i = 0; i < 2; i++)
    t1[i].n.i1 = t3[i].n.i1 = 1;

#pragma omp target map(tofrom : t1, t2, t3)
  for (int i = 0; i < 2; i++) {
    t1[i].n.i3 = t3[i].n.i3 = t1[i].n.i1;
    t1[i].n.i1 = t3[i].n.i1 = 7;
  }

  for (int i = 0; i < 2; i++) {
    if (t1[i].n.i3 != t1[i].n.i1) {
      printf("failed %s. t1[%d].i3 (%d) != t1[%d].i1 (%d)\n", T::name, i,
             t1[i].n.i3, i, t1[i].n.i1);
      return 1;
    }
    if (t3[i].n.i3 != t3[i].n.i1) {
      printf("failed %s. t3[%d].i3 (%d) != t3[%d].i1 (%d)\n", T::name, i,
             t3[i].n.i3, i, t3[i].n.i1);
      return 1;
    }
  }

  int pt0 = omp_target_is_present(&t1[0], omp_get_default_device());
  int pt1 = omp_target_is_present(&t2[1], omp_get_default_device());
  int pt2 = omp_target_is_present(&t3[2], omp_get_default_device());

  printf("present check for %s: t1 %i, t2 %i, t3 %i, expected 3x 0\n", T::name,
         pt0, pt1, pt2);
  return pt0 + pt1 + pt2;
}

int main(int argc, char **argv) {
  int r = 0;
  r += testType<TY1>();
  // CHECK: present check for TY1: t1 0, t2 0, t3 0, expected 3x 0
  r += testType<TY2>();
  // CHECK: present check for TY2: t1 0, t2 0, t3 0, expected 3x 0
  r += testTypeNested<TY3>();
  // CHECK: present check for TY3-ptr: t1 0, t2 0, t3 0, expected 3x 0
  // CHECK: present check for TY3: t1 0, t2 0, t3 0, expected 3x 0
  r += testTypeNested<TY4>();
  // CHECK: present check for TY4-ptr: t1 0, t2 0, t3 0, expected 3x 0
  // CHECK: present check for TY4: t1 0, t2 0, t3 0, expected 3x 0
  return r;
}
