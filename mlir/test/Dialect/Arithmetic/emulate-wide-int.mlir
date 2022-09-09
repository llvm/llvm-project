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

// CHECK-LABEL: func @addi_scalar_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK-NEXT:    [[SUM_L:%.+]], [[CB:%.+]] = arith.addui_carry [[LOW0]], [[LOW1]] : i32, i1
// CHECK-NEXT:    [[CARRY:%.+]]  = arith.extui [[CB]] : i1 to i32
// CHECK-NEXT:    [[SUM_H0:%.+]] = arith.addi [[CARRY]], [[HIGH0]] : i32
// CHECK-NEXT:    [[SUM_H1:%.+]] = arith.addi [[SUM_H0]], [[HIGH1]] : i32
// CHECK:         [[INS0:%.+]]   = vector.insert [[SUM_L]], {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert [[SUM_H1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @addi_scalar_a_b(%a : i64, %b : i64) -> i64 {
    %x = arith.addi %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @addi_vector_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x2xi32>, [[ARG1:%.+]]: vector<4x2xi32>) -> vector<4x2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract_strided_slice [[ARG0]] {offsets = [0, 0], sizes = [4, 1], strides = [1, 1]} : vector<4x2xi32> to vector<4x1xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract_strided_slice [[ARG0]] {offsets = [0, 1], sizes = [4, 1], strides = [1, 1]} : vector<4x2xi32> to vector<4x1xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract_strided_slice [[ARG1]] {offsets = [0, 0], sizes = [4, 1], strides = [1, 1]} : vector<4x2xi32> to vector<4x1xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract_strided_slice [[ARG1]] {offsets = [0, 1], sizes = [4, 1], strides = [1, 1]} : vector<4x2xi32> to vector<4x1xi32>
// CHECK-NEXT:    [[SUM_L:%.+]], [[CB:%.+]] = arith.addui_carry [[LOW0]], [[LOW1]] : vector<4x1xi32>, vector<4x1xi1>
// CHECK-NEXT:    [[CARRY:%.+]]  = arith.extui [[CB]] : vector<4x1xi1> to vector<4x1xi32>
// CHECK-NEXT:    [[SUM_H0:%.+]] = arith.addi [[CARRY]], [[HIGH0]] : vector<4x1xi32>
// CHECK-NEXT:    [[SUM_H1:%.+]] = arith.addi [[SUM_H0]], [[HIGH1]] : vector<4x1xi32>
// CHECK:         [[INS0:%.+]]   = vector.insert_strided_slice [[SUM_L]], {{%.+}} {offsets = [0, 0], strides = [1, 1]} : vector<4x1xi32> into vector<4x2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert_strided_slice [[SUM_H1]], [[INS0]] {offsets = [0, 1], strides = [1, 1]} : vector<4x1xi32> into vector<4x2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<4x2xi32>
func.func @addi_vector_a_b(%a : vector<4xi64>, %b : vector<4xi64>) -> vector<4xi64> {
    %x = arith.addi %a, %b : vector<4xi64>
    return %x : vector<4xi64>
}
