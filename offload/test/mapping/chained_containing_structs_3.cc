// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>

#include <omp.h>

struct R {
  int d;
  int e;
  int f;
};

struct S {
  int a;
  int b;
  struct {
    int c;
    R r;
    R *rp;
  } sub;
  int g;
};

struct T {
  int a;
  int *ptr;
  int b;
};

int main() {
  R r;
  R *rp = new R;
  S s;
  S *sp = new S;
  T t;
  T *tp = new T;

  memset(&r, 0, sizeof(R));
  memset(rp, 0, sizeof(R));
  memset(&s, 0, sizeof(S));
  memset(sp, 0, sizeof(S));
  memset(&t, 0, sizeof(T));
  memset(tp, 0, sizeof(T));

  s.sub.rp = new R;
  sp->sub.rp = new R;

  memset(s.sub.rp, 0, sizeof(R));
  memset(sp->sub.rp, 0, sizeof(R));

  t.ptr = new int[10];
  tp->ptr = new int[10];

  memset(t.ptr, 0, sizeof(int)*10);
  memset(tp->ptr, 0, sizeof(int)*10);

#pragma omp target map(tofrom: r) map(tofrom: r.e)
{
  r.d++;
  r.e += 2;
  r.f += 3;
}
  printf ("%d\n", r.d); // CHECK: 1
  printf ("%d\n", r.e); // CHECK-NEXT: 2
  printf ("%d\n", r.f); // CHECK-NEXT: 3

#pragma omp target map(tofrom: rp[:1]) map(tofrom: rp->e)
{
  rp->d++;
  rp->e += 2;
  rp->f += 3;
}

  printf ("%d\n", rp->d); // CHECK-NEXT: 1
  printf ("%d\n", rp->e); // CHECK-NEXT: 2
  printf ("%d\n", rp->f); // CHECK-NEXT: 3

  int v;
  int *orig_addr_v = &v;
  bool separate_memory_space;

#pragma omp target data map(v)
  {
    void *mapped_ptr_v =
        omp_get_mapped_ptr(orig_addr_v, omp_get_default_device());
    separate_memory_space = mapped_ptr_v != (void*) orig_addr_v;
  }

  const char *mapping_flavour = separate_memory_space ? "separate" : "unified";

#pragma omp target map(to: s) map(tofrom: s.sub.r.e)
{
  s.b++;
  s.sub.r.d+=2;
  s.sub.r.e+=3;
  s.sub.r.f+=4;
}

  printf ("%d/%s\n", s.b, mapping_flavour);
  printf ("%d/%s\n", s.sub.r.d, mapping_flavour);
  printf ("%d/%s\n", s.sub.r.e, mapping_flavour);
  printf ("%d/%s\n", s.sub.r.f, mapping_flavour);

  // CHECK: {{0/separate|1/unified}}
  // CHECK-NEXT: {{0/separate|2/unified}}
  // CHECK-NEXT: 3
  // CHECK-NEXT: {{0/separate|4/unified}}

#pragma omp target map(to: s, s.b) map(to: s.sub.rp[:1]) map(tofrom: s.sub.rp->e)
{
  s.b++;
  s.sub.rp->d+=2;
  s.sub.rp->e+=3;
  s.sub.rp->f+=4;
}

  printf ("%d/%s\n", s.b, mapping_flavour);
  printf ("%d/%s\n", s.sub.rp->d, mapping_flavour);
  printf ("%d/%s\n", s.sub.rp->e, mapping_flavour);
  printf ("%d/%s\n", s.sub.rp->f, mapping_flavour);

  // CHECK-NEXT: {{0/separate|2/unified}}
  // CHECK-NEXT: {{0/separate|2/unified}}
  // CHECK-NEXT: 3
  // CHECK-NEXT: {{0/separate|4/unified}}

#pragma omp target map(to: sp[:1]) map(tofrom: sp->sub.r.e)
{
  sp->b++;
  sp->sub.r.d+=2;
  sp->sub.r.e+=3;
  sp->sub.r.f+=4;
}

  printf ("%d/%s\n", sp->b, mapping_flavour);
  printf ("%d/%s\n", sp->sub.r.d, mapping_flavour);
  printf ("%d/%s\n", sp->sub.r.e, mapping_flavour);
  printf ("%d/%s\n", sp->sub.r.f, mapping_flavour);

  // CHECK-NEXT: {{0/separate|1/unified}}
  // CHECK-NEXT: {{0/separate|2/unified}}
  // CHECK-NEXT: 3
  // CHECK-NEXT: {{0/separate|4/unified}}

#pragma omp target map(to: sp[:1]) map(to: sp->sub.rp[:1]) map(tofrom: sp->sub.rp->e)
{
  sp->b++;
  sp->sub.rp->d+=2;
  sp->sub.rp->e+=3;
  sp->sub.rp->f+=4;
}

  printf ("%d/%s\n", sp->b, mapping_flavour);
  printf ("%d/%s\n", sp->sub.rp->d, mapping_flavour);
  printf ("%d/%s\n", sp->sub.rp->e, mapping_flavour);
  printf ("%d/%s\n", sp->sub.rp->f, mapping_flavour);

  // CHECK-NEXT: {{0/separate|2/unified}}
  // CHECK-NEXT: {{0/separate|2/unified}}
  // CHECK-NEXT: 3
  // CHECK-NEXT: {{0/separate|4/unified}}

#pragma omp target map(tofrom: t) map(tofrom: t.ptr[2:1])
{
  t.a++;
  t.ptr[2]+=2;
  t.b+=3;
}

  printf ("%d\n", t.a); // CHECK-NEXT: 1
  printf ("%d\n", t.ptr[2]); // CHECK-NEXT: 2
  printf ("%d\n", t.b); // CHECK-NEXT: 3

#pragma omp target map(tofrom: t) map(tofrom: t.a)
{
  t.b++;
}

  printf ("%d\n", t.b); // CHECK-NEXT: 4

#pragma omp target map(tofrom: t) map(tofrom: t.ptr[2:1], t.a)
{
  t.a++;
  t.ptr[2]+=2;
  t.b+=3;
}

  printf ("%d\n", t.a); // CHECK-NEXT: 2
  printf ("%d\n", t.ptr[2]); // CHECK-NEXT: 4
  printf ("%d\n", t.b); // CHECK-NEXT: 7

#pragma omp target map(tofrom: t) map(tofrom: t.ptr[2:1], t.a)
{
  /* Empty */
}

  printf ("%d\n", t.a); // CHECK-NEXT: 2
  printf ("%d\n", t.ptr[2]); // CHECK-NEXT: 4
  printf ("%d\n", t.b); // CHECK-NEXT: 7

  delete s.sub.rp;
  delete sp->sub.rp;

  delete[] t.ptr;
  delete[] tp->ptr;

  delete rp;
  delete sp;
  delete tp;

  return 0;
}
