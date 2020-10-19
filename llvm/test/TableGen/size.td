// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Test !size of lists.

// CHECK: --- Defs ---

// CHECK: def A1 {
// CHECK:   int Val = 0;
// CHECK: }

// CHECK: def A2 {
// CHECK:   int Val = 3;
// CHECK: }

// CHECK: def B1 {
// CHECK:   int Val = 0;
// CHECK: }

// CHECK: def B2 {
// CHECK:   int Val = 2;
// CHECK: }

class A<list<int> L> {
  int Val = !size(L);
}

class B<list<string> L> {
  int Val = !size(L);
}

def A1 : A<[]>;
def A2 : A<[1, 1, 2]>;

def B1 : B<[]>;
def B2 : B<["a", "b"]>;

// Test !size of DAGs.

// CHECK: def D0 {
// CHECK:   int Val = 0;
// CHECK: }

// CHECK: def D1 {
// CHECK:   int Val = 1;
// CHECK: }

// CHECK: def D2 {
// CHECK:   int Val = 2;
// CHECK: }

// CHECK: def D3 {
// CHECK:   int Val = 3;
// CHECK: }

// CHECK: def D4 {
// CHECK:   int Val = 4;
// CHECK: }

class D<dag D> {
  int Val = !size(D);
}

def op;

def D0 : D<(op)>;
def D1 : D<(op "string")>;
def D2 : D<(op "string", 42)>;
def D3 : D<(op "string", 42, (op "sub-dag"))>;
def D4 : D<(op "string", 42, (op "sub-dag"), D0.Val)>;

// Test !size of strings.

// CHECK: def S0 {
// CHECK:   int Val = 0;
// CHECK: }

// CHECK: def S1 {
// CHECK:   int Val = 1;
// CHECK: }

// CHECK: def S2 {
// CHECK:   int Val = 2;
// CHECK: }

// CHECK: def S3 {
// CHECK:   int Val = 3;
// CHECK: }

// CHECK: def S4 {
// CHECK:   int Val = 29;
// CHECK: }

class S<string S> {
  int Val = !size(S);
}

def S0 : S<"">;
def S1 : S<"a">;
def S2 : S<"ab">;
def S3 : S<"abc">;
def S4 : S<"This is the end of the world!">;

