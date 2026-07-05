// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: --- Defs ---
// CHECK: def Op1 {       // Op
// CHECK-NEXT:  string res = "yes";
// CHECK-NEXT: }
// CHECK: def Op2 {       // Op
// CHECK-NEXT:  string res = "no";
// CHECK-NEXT: }

class A<int a> {
  int x = a;
}

class B<int a> : A<a> {
  bit y = 0;
}

class C<int a> {
  int z = !add(a, 16);
}

class D<int a> : B<a>, C<a>;

def E1 : D<5>;
def E2 : B<2>;

class Op<A value> {
  string res = !if(!isa<C>(value), "yes", "no");
}

def Op1 : Op<E1>;
def Op2 : Op<E2>;
