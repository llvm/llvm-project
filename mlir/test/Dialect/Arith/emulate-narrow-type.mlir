// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=8" %s | FileCheck %s

// Expect no conversions, f32 is not an integer type.
// CHECK-LABEL: func @identity_f32
// CHECK-SAME:    ([[ARG:%.+]]: f32) -> f32
// CHECK-NEXT:    return [[ARG]] : f32
func.func @identity_f32(%a : f32) -> f32 {
    return %a : f32
}

// Expect no conversions, i32 is supported.
// CHECK-LABEL: func @identity_i32
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    return [[ARG]] : vector<2xi32>
func.func @identity_i32(%a : vector<2xi32>) -> vector<2xi32> {
    return %a : vector<2xi32>
}

// CHECK-LABEL: func @identity_scalar
// CHECK-SAME:     ([[ARG:%.+]]: i8) -> i8
// CHECK-NEXT:     return [[ARG]] : i8
func.func @identity_scalar(%x : i4) -> i4 {
    return %x : i4
}

// CHECK-LABEL: func @identity_vector
// CHECK-SAME:     ([[ARG:%.+]]: vector<4xi8>) -> vector<4xi8>
// CHECK-NEXT:     return [[ARG]] : vector<4xi8>
func.func @identity_vector(%x : vector<4xi4>) -> vector<4xi4> {
    return %x : vector<4xi4>
}

// CHECK-LABEL: func @identity_vector2d
// CHECK-SAME:     ([[ARG:%.+]]: vector<3x4xi8>) -> vector<3x4xi8>
// CHECK-NEXT:     return [[ARG]] : vector<3x4xi8>
func.func @identity_vector2d(%x : vector<3x4xi4>) -> vector<3x4xi4> {
    return %x : vector<3x4xi4>
}

// CHECK-LABEL: func @call
// CHECK-SAME:     ([[ARG:%.+]]: vector<4xi8>) -> vector<4xi8>
// CHECK-NEXT:     [[RES:%.+]] = call @identity_vector([[ARG]]) : (vector<4xi8>) -> vector<4xi8>
// CHECK-NEXT:     return [[RES]] : vector<4xi8>
func.func @call(%a : vector<4xi4>) -> vector<4xi4> {
    %res = func.call @identity_vector(%a) : (vector<4xi4>) -> vector<4xi4>
    return %res : vector<4xi4>
}
