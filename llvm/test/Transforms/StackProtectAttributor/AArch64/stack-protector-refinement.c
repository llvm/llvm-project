// RUN: clang -O2 -fstack-protector-strong -emit-llvm -S  %s -o - | FileCheck %s

__attribute__((noinline))
int foo1(int *p) {
  return p[10];
}
int bar1() {
// CHECK-LABEL: define {{[^@]+}}@bar1
// CHECK-SAME: () local_unnamed_addr #[[ATTR1:[0-9]+]] {
  int x[128];
  return foo1(x);
}

__attribute__((noinline))
int foo2(int *p) {
  return p[1000];
}
int bar2() {
// CHECK-LABEL: define {{[^@]+}}@bar2
// CHECK-SAME: () local_unnamed_addr #[[ATTR2:[0-9]+]] {
  int x[128];
  return foo2(x);
}

int k;
__attribute__((noinline))
int foo3(int *p) {
  return p[k];
}
int bar3() {
// CHECK-LABEL: define {{[^@]+}}@bar3
// CHECK-SAME: () local_unnamed_addr #[[ATTR3:[0-9]+]] {
  int x[128];
  return foo3(x);
}

__attribute__((noinline))
int foo4(int *p);
int bar4() {
// CHECK-LABEL: define {{[^@]+}}@bar4
// CHECK-SAME: () local_unnamed_addr #[[ATTR4:[0-9]+]] {
  int x[128];
  return foo4(x);
}

int bar5() {
// CHECK-LABEL: define {{[^@]+}}@bar5
// CHECK-SAME: () local_unnamed_addr #[[ATTR5:[0-9]+]] {
  int x[128];
  int i;
  for (i = 0; i < 128; ++i)
    x[i] = i;
  return foo1(x);
}

//.
// CHECK: attributes #[[ATTR1]] =
// CHECK-NOT: sspstrong
// CHECK: attributes #[[ATTR2]] = {{.* sspstrong}}
// CHECK: attributes #[[ATTR3]] = {{.* sspstrong}}
// CHECK: attributes #[[ATTR4]] = {{.* sspstrong}}
// CHECK: attributes #[[ATTR5]] =
// CHECK-NOT: sspstrong
//.