// RUN: mlir-opt --arith-emulate-wide-int="widest-int-supported=32" %s | FileCheck %s

// Expect no conversions, i32 is supported.
// CHECK-LABEL: func @addi_same_i32
// CHECK-SAME:    ([[ARG:%.+]]: i32) -> i32
// CHECK-NEXT:    [[X:%.+]] = arith.addi [[ARG]], [[ARG]] : i32
// CHECK-NEXT:    return [[X]] : i32
func.func @addi_same_i32(%a : i32) -> i32 {
    %x = arith.addi %a, %a : i32
    return %x : i32
}

// Expect no conversions, i32 is supported.
// CHECK-LABEL: func @addi_same_vector_i32
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[X:%.+]] = arith.addi [[ARG]], [[ARG]] : vector<2xi32>
// CHECK-NEXT:    return [[X]] : vector<2xi32>
func.func @addi_same_vector_i32(%a : vector<2xi32>) -> vector<2xi32> {
    %x = arith.addi %a, %a : vector<2xi32>
    return %x : vector<2xi32>
}

// CHECK-LABEL: func @identity_scalar
// CHECK-SAME:     ([[ARG:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:     return [[ARG]] : vector<2xi32>
func.func @identity_scalar(%x : i64) -> i64 {
    return %x : i64
}

// CHECK-LABEL: func @identity_vector
// CHECK-SAME:     ([[ARG:%.+]]: vector<4x2xi32>) -> vector<4x2xi32>
// CHECK-NEXT:     return [[ARG]] : vector<4x2xi32>
func.func @identity_vector(%x : vector<4xi64>) -> vector<4xi64> {
    return %x : vector<4xi64>
}

// CHECK-LABEL: func @identity_vector2d
// CHECK-SAME:     ([[ARG:%.+]]: vector<3x4x2xi32>) -> vector<3x4x2xi32>
// CHECK-NEXT:     return [[ARG]] : vector<3x4x2xi32>
func.func @identity_vector2d(%x : vector<3x4xi64>) -> vector<3x4xi64> {
    return %x : vector<3x4xi64>
}

// CHECK-LABEL: func @call
// CHECK-SAME:     ([[ARG:%.+]]: vector<4x2xi32>) -> vector<4x2xi32>
// CHECK-NEXT:     [[RES:%.+]] = call @identity_vector([[ARG]]) : (vector<4x2xi32>) -> vector<4x2xi32>
// CHECK-NEXT:     return [[RES]] : vector<4x2xi32>
func.func @call(%a : vector<4xi64>) -> vector<4xi64> {
    %res = func.call @identity_vector(%a) : (vector<4xi64>) -> vector<4xi64>
    return %res : vector<4xi64>
}

// CHECK-LABEL: func @constant_scalar
// CHECK-SAME:     () -> vector<2xi32>
// CHECK-NEXT:     [[C0:%.+]] = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:     [[C1:%.+]] = arith.constant dense<[0, 1]> : vector<2xi32>
// CHECK-NEXT:     [[C2:%.+]] = arith.constant dense<[-7, -1]> : vector<2xi32>
// CHECK-NEXT:     return [[C0]] : vector<2xi32>
func.func @constant_scalar() -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 4294967296 : i64
    %c2 = arith.constant -7 : i64
    return %c0 : i64
}

// CHECK-LABEL: func @constant_vector
// CHECK-SAME:     () -> vector<3x2xi32>
// CHECK-NEXT:     [[C0:%.+]] = arith.constant dense
// CHECK-SAME{LITERAL}:                             <[[0, 1], [0, 1], [0, 1]]> : vector<3x2xi32>
// CHECK-NEXT:     [[C1:%.+]] = arith.constant dense
// CHECK-SAME{LITERAL}:                             <[[0, 0], [1, 0], [-2, -1]]> : vector<3x2xi32>
// CHECK-NEXT:     return [[C0]] : vector<3x2xi32>
func.func @constant_vector() -> vector<3xi64> {
    %c0 = arith.constant dense<4294967296> : vector<3xi64>
    %c1 = arith.constant dense<[0, 1, -2]> : vector<3xi64>
    return %c0 : vector<3xi64>
}
