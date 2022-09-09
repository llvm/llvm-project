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
