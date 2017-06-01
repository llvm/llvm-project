// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

def shifts {
    bits<2> b = 0b10;
    int i = 2;
    int shifted_b = !shl(b, 2);
    int shifted_i = !shl(i, 2);
}
// CHECK: def shifts
// CHECK: shifted_b = 8
// CHECK: shifted_i = 8

class Int<int value> {
  int Value = value;
}

def v1022   : Int<1022>;

// CHECK: def v0
// CHECK: Value = 0

// CHECK: def v1
// CHECK: Value = 1

// CHECK: def v1023
// CHECK: Value = 1023
def v1023 : Int<!or(v1022.Value, 1)>;

def v1024   : Int<1024>;
// CHECK: def v1024
// CHECK: Value = 1024

def v1025   : Int<!add(v1024.Value, 1)>;
// CHECK: def v1025
// CHECK: Value = 1025

def v2048   : Int<!add(v1024.Value, v1024.Value)>;
// CHECK: def v2048
// CHECK: Value = 2048

def v0 : Int<!and(v1024.Value, v2048.Value)>;
def v1 : Int<!and(v1025.Value, 1)>;

// CHECK: def v3072
// CHECK: Value = 3072
def v3072 : Int<!or(v1024.Value, v2048.Value)>;
