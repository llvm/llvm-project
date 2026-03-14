// RUN: llvm-tblgen --no-warn-on-unused-template-args %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def X0 {
// CHECK:   int ret = 0;
// CHECK: }

// CHECK: def X1 {
// CHECK:   int ret = 1;
// CHECK: }

// CHECK: def Y0 {
// CHECK:   int ret = 0;
// CHECK: }

// CHECK: def Y1 {
// CHECK:   int ret = 11;
// CHECK: }

class A<int dummy>;
class B<int num> : A<num> {
  int Num = num;
}

class X<A a> {
  int ret = !isa<B>(a);
}

class Y<A a> {
  int ret = !if(!isa<B>(a), !cast<B>(a).Num, 0);
}

def X0 : X<A<0>>;
def X1 : X<B<0>>;

def Y0 : Y<A<10>>;
def Y1 : Y<B<11>>;
