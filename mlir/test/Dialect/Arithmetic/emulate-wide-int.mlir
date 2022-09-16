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

// CHECK-LABEL: func @extsi_scalar
// CHECK-SAME:    ([[ARG:%.+]]: i16) -> vector<2xi32>
// CHECK-NEXT:    [[EXT:%.+]]  = arith.extsi [[ARG]] : i16 to i32
// CHECK-NEXT:    [[SZ:%.+]]   = arith.constant 0 : i32
// CHECK-NEXT:    [[SB:%.+]]   = arith.cmpi slt, [[EXT]], [[SZ]] : i32
// CHECK-NEXT:    [[SV:%.+]]   = arith.extsi [[SB]] : i1 to i32
// CHECK-NEXT:    [[VZ:%.+]]   = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]] = vector.insert [[EXT]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]] = vector.insert [[SV]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK:         return [[INS1]] : vector<2xi32>
func.func @extsi_scalar(%a : i16) -> i64 {
    %r = arith.extsi %a : i16 to i64
    return %r : i64
}

// CHECK-LABEL: func @extsi_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3xi16>) -> vector<3x2xi32>
// CHECK-NEXT:    [[SHAPE:%.+]] = vector.shape_cast [[ARG]] : vector<3xi16> to vector<3x1xi16>
// CHECK-NEXT:    [[EXT:%.+]]   = arith.extsi [[SHAPE]] : vector<3x1xi16> to vector<3x1xi32>
// CHECK-NEXT:    [[CSTE:%.+]]  = arith.constant dense<0> : vector<3x1xi32>
// CHECK-NEXT:    [[CMP:%.+]]   = arith.cmpi slt, [[EXT]], [[CSTE]] : vector<3x1xi32>
// CHECK-NEXT:    [[HIGH:%.+]]  = arith.extsi [[CMP]] : vector<3x1xi1> to vector<3x1xi32>
// CHECK-NEXT:    [[CSTZ:%.+]]  = arith.constant dense<0> : vector<3x2xi32>
// CHECK-NEXT:    [[INS0:%.+]]  = vector.insert_strided_slice [[EXT]], [[CSTZ]] {offsets = [0, 0], strides = [1, 1]} : vector<3x1xi32> into vector<3x2xi32>
// CHECK-NEXT:    [[INS1:%.+]]  = vector.insert_strided_slice [[HIGH]], [[INS0]] {offsets = [0, 1], strides = [1, 1]} : vector<3x1xi32> into vector<3x2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<3x2xi32>
func.func @extsi_vector(%a : vector<3xi16>) -> vector<3xi64> {
    %r = arith.extsi %a : vector<3xi16> to vector<3xi64>
    return %r : vector<3xi64>
}

// CHECK-LABEL: func @extui_scalar1
// CHECK-SAME:    ([[ARG:%.+]]: i16) -> vector<2xi32>
// CHECK-NEXT:    [[EXT:%.+]]  = arith.extui [[ARG]] : i16 to i32
// CHECK-NEXT:    [[VZ:%.+]]   = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]] = vector.insert [[EXT]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK:         return [[INS0]] : vector<2xi32>
func.func @extui_scalar1(%a : i16) -> i64 {
    %r = arith.extui %a : i16 to i64
    return %r : i64
}

// CHECK-LABEL: func @extui_scalar2
// CHECK-SAME:    ([[ARG:%.+]]: i32) -> vector<2xi32>
// CHECK-NEXT:    [[VZ:%.+]]   = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]] = vector.insert [[ARG]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK:         return [[INS0]] : vector<2xi32>
func.func @extui_scalar2(%a : i32) -> i64 {
    %r = arith.extui %a : i32 to i64
    return %r : i64
}

// CHECK-LABEL: func @extui_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3xi16>) -> vector<3x2xi32>
// CHECK-NEXT:    [[SHAPE:%.+]] = vector.shape_cast [[ARG]] : vector<3xi16> to vector<3x1xi16>
// CHECK-NEXT:    [[EXT:%.+]]   = arith.extui [[SHAPE]] : vector<3x1xi16> to vector<3x1xi32>
// CHECK-NEXT:    [[CST:%.+]]   = arith.constant dense<0> : vector<3x2xi32>
// CHECK-NEXT:    [[INS0:%.+]]  = vector.insert_strided_slice [[EXT]], [[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<3x1xi32> into vector<3x2xi32>
// CHECK:         return [[INS0]] : vector<3x2xi32>
func.func @extui_vector(%a : vector<3xi16>) -> vector<3xi64> {
    %r = arith.extui %a : vector<3xi16> to vector<3xi64>
    return %r : vector<3xi64>
}

// CHECK-LABEL: func @trunci_scalar1
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> i32
// CHECK-NEXT:    [[EXT:%.+]] = vector.extract [[ARG]][0] : vector<2xi32>
// CHECK-NEXT:    return [[EXT]] : i32
func.func @trunci_scalar1(%a : i64) -> i32 {
    %b = arith.trunci %a : i64 to i32
    return %b : i32
}

// CHECK-LABEL: func @trunci_scalar2
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> i16
// CHECK-NEXT:    [[EXTR:%.+]] = vector.extract [[ARG]][0] : vector<2xi32>
// CHECK-NEXT:    [[TRNC:%.+]] = arith.trunci [[EXTR]] : i32 to i16
// CHECK-NEXT:    return [[TRNC]] : i16
func.func @trunci_scalar2(%a : i64) -> i16 {
    %b = arith.trunci %a : i64 to i16
    return %b : i16
}

// CHECK-LABEL: func @trunci_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3x2xi32>) -> vector<3xi16>
// CHECK-NEXT:    [[EXTR:%.+]]  = vector.extract_strided_slice [[ARG]] {offsets = [0, 0], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[SHAPE:%.+]] = vector.shape_cast [[EXTR]] : vector<3x1xi32> to vector<3xi32>
// CHECK-NEXT:    [[TRNC:%.+]]  = arith.trunci [[SHAPE]] : vector<3xi32> to vector<3xi16>
// CHECK-NEXT:    return [[TRNC]] : vector<3xi16>
func.func @trunci_vector(%a : vector<3xi64>) -> vector<3xi16> {
    %b = arith.trunci %a : vector<3xi64> to vector<3xi16>
    return %b : vector<3xi16>
}

// CHECK-LABEL: func.func @muli_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]      = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]     = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]      = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]     = vector.extract [[ARG1]][1] : vector<2xi32>
//
// CHECK-DAG:     [[MASK:%.+]]      = arith.constant 65535 : i32
// CHECK-DAG:     [[C16:%.+]]       = arith.constant 16 : i32
//
// CHECK:         [[LOWLOW0:%.+]]   = arith.andi [[LOW0]], [[MASK]] : i32
// CHECK-NEXT:    [[HIGHLOW0:%.+]]  = arith.shrui [[LOW0]], [[C16]] : i32
// CHECK-NEXT:    [[LOWHIGH0:%.+]]  = arith.andi [[HIGH0]], [[MASK]] : i32
// CHECK-NEXT:    [[HIGHHIGH0:%.+]] = arith.shrui [[HIGH0]], [[C16]] : i32
// CHECK-NEXT:    [[LOWLOW1:%.+]]   = arith.andi [[LOW1]], [[MASK]] : i32
// CHECK-NEXT:    [[HIGHLOW1:%.+]]  = arith.shrui [[LOW1]], [[C16]] : i32
// CHECK-NEXT:    [[LOWHIGH1:%.+]]  = arith.andi [[HIGH1]], [[MASK]] : i32
// CHECK-NEXT:    [[HIGHHIGH1:%.+]] = arith.shrui [[HIGH1]], [[C16]] : i32
//
// CHECK-DAG:     {{%.+}}           = arith.muli [[LOWLOW0]], [[LOWLOW1]] : i32
// CHECK-DAG      {{%.+}}           = arith.muli [[LOWLOW0]], [[HIGHLOW1]] : i32
// CHECK-DAG:     {{%.+}}           = arith.muli [[LOWLOW0]], [[LOWHIGH1]] : i32
// CHECK-DAG:     {{%.+}}           = arith.muli [[LOWLOW0]], [[HIGHHIGH1]] : i32
//
// CHECK-DAG:     {{%.+}}           = arith.muli [[HIGHLOW0]], [[LOWLOW1]] : i32
// CHECK-DAG:     {{%.+}}           = arith.muli [[HIGHLOW0]], [[HIGHLOW1]] : i32
// CHECK-DAG:     {{%.+}}           = arith.muli [[HIGHLOW0]], [[LOWHIGH1]] : i32
//
// CHECK-DAG:     {{%.+}}           = arith.muli [[LOWHIGH0]], [[LOWLOW1]] : i32
// CHECK-DAG:     {{%.+}}           = arith.muli [[LOWHIGH0]], [[HIGHLOW1]] : i32
//
// CHECK-DAG:     {{%.+}}           = arith.muli [[HIGHHIGH0]], [[LOWLOW1]] : i32
//
// CHECK:         [[RESHIGH0:%.+]]  = arith.shli {{%.+}}, [[C16]] : i32
// CHECK-NEXT:    [[RES0:%.+]]      = arith.ori {{%.+}}, [[RESHIGH0]] : i32
// CHECK-NEXT:    [[RESHIGH1:%.+]]  = arith.shli {{%.+}}, [[C16]] : i32
// CHECK-NEXT:    [[RES1:%.+]]      = arith.ori {{%.+}}, [[RESHIGH1]] : i32
// CHECK-NEXT:    [[VZ:%.+]]        = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]]      = vector.insert [[RES0]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]      = vector.insert [[RES1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @muli_scalar(%a : i64, %b : i64) -> i64 {
    %m = arith.muli %a, %b : i64
    return %m : i64
}

// CHECK-LABEL: func.func @muli_vector
// CHECK-SAME:    ({{%.+}}: vector<3x2xi32>, {{%.+}}: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:       return {{%.+}} : vector<3x2xi32>
func.func @muli_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %m = arith.muli %a, %b : vector<3xi64>
    return %m : vector<3xi64>
}
