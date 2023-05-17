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

// Expect no conversions, index is not sized.
// CHECK-LABEL: func @addi_same_index
// CHECK-SAME:    ([[ARG:%.+]]: index) -> index
// CHECK-NEXT:    [[X:%.+]] = arith.addi [[ARG]], [[ARG]] : index
// CHECK-NEXT:    return [[X]] : index
func.func @addi_same_index(%a : index) -> index {
    %x = arith.addi %a, %a : index
    return %x : index
}

// Expect no conversions, f64 is not an integer type.
// CHECK-LABEL: func @identity_f64
// CHECK-SAME:    ([[ARG:%.+]]: f64) -> f64
// CHECK-NEXT:    return [[ARG]] : f64
func.func @identity_f64(%a : f64) -> f64 {
    return %a : f64
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
// CHECK-NEXT:    [[SUM_L:%.+]], [[CB:%.+]] = arith.addui_extended [[LOW0]], [[LOW1]] : i32, i1
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
// CHECK-NEXT:    [[SUM_L:%.+]], [[CB:%.+]] = arith.addui_extended [[LOW0]], [[LOW1]] : vector<4x1xi32>, vector<4x1xi1>
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

// CHECK-LABEL: func.func @cmpi_eq_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK-NEXT:    [[LHSLOW:%.+]]  = vector.extract [[LHS]][0] : vector<2xi32>
// CHECK-NEXT:    [[LHSHIGH:%.+]] = vector.extract [[LHS]][1] : vector<2xi32>
// CHECK-NEXT:    [[RHSLOW:%.+]]  = vector.extract [[RHS]][0] : vector<2xi32>
// CHECK-NEXT:    [[RHSHIGH:%.+]] = vector.extract [[RHS]][1] : vector<2xi32>
// CHECK-NEXT:    [[CLOW:%.+]]  = arith.cmpi eq, [[LHSLOW]], [[RHSLOW]] : i32
// CHECK-NEXT:    [[CHIGH:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.andi [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_eq_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi eq, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_eq_vector
// CHECK-SAME:    ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3xi1>
// CHECK-NEXT:    [[LOW0:%.+]]  = vector.extract_strided_slice [[ARG0]] {offsets = [0, 0], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[HIGH0:%.+]] = vector.extract_strided_slice [[ARG0]] {offsets = [0, 1], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[LOW1:%.+]]  = vector.extract_strided_slice [[ARG1]] {offsets = [0, 0], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[HIGH1:%.+]] = vector.extract_strided_slice [[ARG1]] {offsets = [0, 1], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[CLOW:%.+]]  = arith.cmpi eq, [[LOW0]], [[LOW1]] : vector<3x1xi32>
// CHECK-NEXT:    [[CHIGH:%.+]] = arith.cmpi eq, [[HIGH0]], [[HIGH1]] : vector<3x1xi32>
// CHECK-NEXT:    [[RES:%.+]]   = arith.andi [[CLOW]], [[CHIGH]] : vector<3x1xi1>
// CHECK-NEXT:    [[CAST:%.+]]  = vector.shape_cast [[RES]] : vector<3x1xi1> to vector<3xi1>
// CHECK:         return [[CAST]] : vector<3xi1>
func.func @cmpi_eq_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi1> {
    %r = arith.cmpi eq, %a, %b : vector<3xi64>
    return %r : vector<3xi1>
}

// CHECK-LABEL: func.func @cmpi_ne_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK-NEXT:    [[LHSLOW:%.+]]  = vector.extract [[LHS]][0] : vector<2xi32>
// CHECK-NEXT:    [[LHSHIGH:%.+]] = vector.extract [[LHS]][1] : vector<2xi32>
// CHECK-NEXT:    [[RHSLOW:%.+]]  = vector.extract [[RHS]][0] : vector<2xi32>
// CHECK-NEXT:    [[RHSHIGH:%.+]] = vector.extract [[RHS]][1] : vector<2xi32>
// CHECK-NEXT:    [[CLOW:%.+]]  = arith.cmpi ne, [[LHSLOW]], [[RHSLOW]] : i32
// CHECK-NEXT:    [[CHIGH:%.+]] = arith.cmpi ne, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.ori [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_ne_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi ne, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_ne_vector
// CHECK-SAME:    ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3xi1>
// CHECK:         [[CLOW:%.+]]  = arith.cmpi ne, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    [[CHIGH:%.+]] = arith.cmpi ne, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    [[RES:%.+]]   = arith.ori [[CLOW]], [[CHIGH]] : vector<3x1xi1>
// CHECK-NEXT:    [[CAST:%.+]]  = vector.shape_cast [[RES]] : vector<3x1xi1> to vector<3xi1>
// CHECK:         return [[CAST]] : vector<3xi1>
func.func @cmpi_ne_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi1> {
    %r = arith.cmpi ne, %a, %b : vector<3xi64>
    return %r : vector<3xi1>
}

// CHECK-LABEL: func.func @cmpi_sge_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK-NEXT:    [[LHSLOW:%.+]]  = vector.extract [[LHS]][0] : vector<2xi32>
// CHECK-NEXT:    [[LHSHIGH:%.+]] = vector.extract [[LHS]][1] : vector<2xi32>
// CHECK-NEXT:    [[RHSLOW:%.+]]  = vector.extract [[RHS]][0] : vector<2xi32>
// CHECK-NEXT:    [[RHSHIGH:%.+]] = vector.extract [[RHS]][1] : vector<2xi32>
// CHECK-NEXT:    [[CLOW:%.+]]   = arith.cmpi uge, [[LHSLOW]], [[RHSLOW]] : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi sge, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_sge_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi sge, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_sge_vector
// CHECK-SAME:    ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3xi1>
// CHECK:         [[CLOW:%.+]]   = arith.cmpi uge, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         [[CHIGH:%.+]]  = arith.cmpi sge, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    [[RES:%.+]]    = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : vector<3x1xi1>
// CHECK-NEXT:    [[CAST:%.+]]   = vector.shape_cast [[RES]] : vector<3x1xi1> to vector<3xi1>
// CHECK:         return [[CAST]] : vector<3xi1>
func.func @cmpi_sge_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi1> {
    %r = arith.cmpi sge, %a, %b : vector<3xi64>
    return %r : vector<3xi1>
}

// CHECK-LABEL: func.func @cmpi_sgt_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi ugt, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi sgt, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_sgt_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi sgt, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_sle_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi ule, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi sle, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_sle_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi sle, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_slt_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi ult, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi slt, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_slt_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi slt, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_uge_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi uge, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi uge, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_uge_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi uge, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_ugt_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi ugt, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi ugt, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_ugt_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi ugt, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_ule_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi ule, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi ule, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_ule_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi ule, %a, %b : i64
    return %r : i1
}

// CHECK-LABEL: func.func @cmpi_ult_scalar
// CHECK-SAME:    ([[LHS:%.+]]: vector<2xi32>, [[RHS:%.+]]: vector<2xi32>)
// CHECK:         [[CLOW:%.+]]   = arith.cmpi ult, {{%.+}}, {{%.+}} : i32
// CHECK-NEXT:    [[CHIGH:%.+]]  = arith.cmpi ult, [[LHSHIGH:%.+]], [[RHSHIGH:%.+]] : i32
// CHECK-NEXT:    [[HIGHEQ:%.+]] = arith.cmpi eq, [[LHSHIGH]], [[RHSHIGH]] : i32
// CHECK-NEXT:    [[RES:%.+]]   = arith.select [[HIGHEQ]], [[CLOW]], [[CHIGH]] : i1
// CHECK:         return [[RES]] : i1
func.func @cmpi_ult_scalar(%a : i64, %b : i64) -> i1 {
    %r = arith.cmpi ult, %a, %b : i64
    return %r : i1
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

// CHECK-LABEL: func @index_cast_int_to_index_scalar
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> index
// CHECK-NEXT:    [[EXT:%.+]]  = vector.extract [[ARG]][0] : vector<2xi32>
// CHECK-NEXT:    [[RES:%.+]]  = arith.index_cast [[EXT]] : i32 to index
// CHECK-NEXT:    return [[RES]] : index
func.func @index_cast_int_to_index_scalar(%a : i64) -> index {
    %r = arith.index_cast %a : i64 to index
    return %r : index
}

// CHECK-LABEL: func @index_cast_int_to_index_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3x2xi32>) -> vector<3xindex>
// CHECK-NEXT:    [[EXT:%.+]]   = vector.extract_strided_slice [[ARG]] {offsets = [0, 0], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[SHAPE:%.+]] = vector.shape_cast [[EXT]] : vector<3x1xi32> to vector<3xi32>
// CHECK-NEXT:    [[RES:%.+]]   = arith.index_cast [[SHAPE]] : vector<3xi32> to vector<3xindex>
// CHECK-NEXT:    return [[RES]] : vector<3xindex>
func.func @index_cast_int_to_index_vector(%a : vector<3xi64>) -> vector<3xindex> {
    %r = arith.index_cast %a : vector<3xi64> to vector<3xindex>
    return %r : vector<3xindex>
}

// CHECK-LABEL: func @index_castui_int_to_index_scalar
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> index
// CHECK-NEXT:    [[EXT:%.+]]  = vector.extract [[ARG]][0] : vector<2xi32>
// CHECK-NEXT:    [[RES:%.+]]  = arith.index_castui [[EXT]] : i32 to index
// CHECK-NEXT:    return [[RES]] : index
func.func @index_castui_int_to_index_scalar(%a : i64) -> index {
    %r = arith.index_castui %a : i64 to index
    return %r : index
}

// CHECK-LABEL: func @index_castui_int_to_index_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3x2xi32>) -> vector<3xindex>
// CHECK-NEXT:    [[EXT:%.+]]   = vector.extract_strided_slice [[ARG]] {offsets = [0, 0], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[SHAPE:%.+]] = vector.shape_cast [[EXT]] : vector<3x1xi32> to vector<3xi32>
// CHECK-NEXT:    [[RES:%.+]]   = arith.index_castui [[SHAPE]] : vector<3xi32> to vector<3xindex>
// CHECK-NEXT:    return [[RES]] : vector<3xindex>
func.func @index_castui_int_to_index_vector(%a : vector<3xi64>) -> vector<3xindex> {
    %r = arith.index_castui %a : vector<3xi64> to vector<3xindex>
    return %r : vector<3xindex>
}

// CHECK-LABEL: func @index_cast_index_to_int_scalar
// CHECK-SAME:    ([[ARG:%.+]]: index) -> vector<2xi32>
// CHECK-NEXT:    [[CAST:%.+]]  = arith.index_cast [[ARG]] : index to i32
// CHECK-NEXT:    [[C0I32:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:    [[NEG:%.+]]   = arith.cmpi slt, [[CAST]], [[C0I32]] : i32
// CHECK-NEXT:    [[EXT:%.+]]   = arith.extsi [[NEG]] : i1 to i32
// CHECK-NEXT:    [[VZ:%.+]]    = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]]  = vector.insert [[CAST]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]  = vector.insert [[EXT]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @index_cast_index_to_int_scalar(%a : index) -> i64 {
    %r = arith.index_cast %a : index to i64
    return %r : i64
}

// CHECK-LABEL: func @index_cast_index_to_int_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3xindex>) -> vector<3x2xi32>
// CHECK-NEXT:    arith.index_cast [[ARG]] : vector<3xindex> to vector<3xi32>
// CHECK-NEXT:    vector.shape_cast
// CHECK-NEXT:    arith.constant dense<0> : vector<3x1xi32>
// CHECK-NEXT:    arith.cmpi slt
// CHECK-NEXT:    arith.extsi
// CHECK-NEXT:    arith.constant dense<0> : vector<3x2xi32>
// CHECK-NEXT:    vector.insert_strided_slice
// CHECK-NEXT:    vector.insert_strided_slice
// CHECK-NEXT:    return {{%.+}} : vector<3x2xi32>
func.func @index_cast_index_to_int_vector(%a : vector<3xindex>) -> vector<3xi64> {
    %r = arith.index_cast %a : vector<3xindex> to vector<3xi64>
    return %r : vector<3xi64>
}

// CHECK-LABEL: func @index_castui_index_to_int_scalar
// CHECK-SAME:    ([[ARG:%.+]]: index) -> vector<2xi32>
// CHECK-NEXT:    [[CAST:%.+]]  = arith.index_castui [[ARG]] : index to i32
// CHECK-NEXT:    [[VZ:%.+]]    = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.insert [[CAST]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[RES]] : vector<2xi32>
func.func @index_castui_index_to_int_scalar(%a : index) -> i64 {
    %r = arith.index_castui %a : index to i64
    return %r : i64
}

// CHECK-LABEL: func @index_castui_index_to_int_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3xindex>) -> vector<3x2xi32>
// CHECK-NEXT:    [[CAST:%.+]]  = arith.index_castui [[ARG]] : vector<3xindex> to vector<3xi32>
// CHECK-NEXT:    [[SHAPE:%.+]] = vector.shape_cast [[CAST]] : vector<3xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[CST:%.+]]   = arith.constant dense<0> : vector<3x2xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.insert_strided_slice [[SHAPE]], [[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<3x1xi32> into vector<3x2xi32>
// CHECK-NEXT:    return [[RES]] : vector<3x2xi32>
func.func @index_castui_index_to_int_vector(%a : vector<3xindex>) -> vector<3xi64> {
    %r = arith.index_castui %a : vector<3xindex> to vector<3xi64>
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

// CHECK-LABEL: func @maxui_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         [[INS0:%.+]]   = vector.insert {{%.+}}, {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert {{%.+}}, [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @maxui_scalar(%a : i64, %b : i64) -> i64 {
    %x = arith.maxui %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @maxui_vector
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @maxui_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.maxui %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func @maxsi_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi sgt
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         [[INS0:%.+]]   = vector.insert {{%.+}}, {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert {{%.+}}, [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @maxsi_scalar(%a : i64, %b : i64) -> i64 {
    %x = arith.maxsi %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @maxsi_vector
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi sgt
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @maxsi_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.maxsi %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func @minui_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         [[INS0:%.+]]   = vector.insert {{%.+}}, {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert {{%.+}}, [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @minui_scalar(%a : i64, %b : i64) -> i64 {
    %x = arith.minui %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @minui_vector
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @minui_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.minui %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func @minsi_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi slt
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         [[INS0:%.+]]   = vector.insert {{%.+}}, {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert {{%.+}}, [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @minsi_scalar(%a : i64, %b : i64) -> i64 {
    %x = arith.minsi %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @minsi_vector
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi slt
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @minsi_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.minsi %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func.func @select_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>, [[ARG2:%.+]]: i1)
// CHECK-SAME:    -> vector<2xi32>
// CHECK-NEXT:    [[TLOW:%.+]] = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[THIGH:%.+]] = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[FLOW:%.+]] = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[FHIGH:%.+]] = vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK-NEXT:    [[SLOW:%.+]] = arith.select [[ARG2]], [[TLOW]], [[FLOW]] : i32
// CHECK-NEXT:    [[SHIGH:%.+]] = arith.select [[ARG2]], [[THIGH]], [[FHIGH]] : i32
// CHECK-NEXT:    [[VZ:%.+]]   = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]] = vector.insert [[SLOW]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]] = vector.insert [[SHIGH]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK:         return [[INS1]] : vector<2xi32>
func.func @select_scalar(%a : i64, %b : i64, %c : i1) -> i64 {
    %r = arith.select %c, %a, %b : i64
    return %r : i64
}

// CHECK-LABEL: func.func @select_vector_whole
// CHECK-SAME:    ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>, [[ARG2:%.+]]: i1)
// CHECK-SAME:    -> vector<3x2xi32>
// CHECK:         arith.select {{%.+}}, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    arith.select {{%.+}}, {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         return {{%.+}} : vector<3x2xi32>
func.func @select_vector_whole(%a : vector<3xi64>, %b : vector<3xi64>, %c : i1) -> vector<3xi64> {
    %r = arith.select %c, %a, %b : vector<3xi64>
    return %r : vector<3xi64>
}

// CHECK-LABEL: func.func @select_vector_elementwise
// CHECK-SAME:    ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>, [[ARG2:%.+]]: vector<3xi1>)
// CHECK-SAME:    -> vector<3x2xi32>
// CHECK:         arith.select {{%.+}}, {{%.+}}, {{%.+}} : vector<3x1xi1>, vector<3x1xi32>
// CHECK-NEXT:    arith.select {{%.+}}, {{%.+}}, {{%.+}} : vector<3x1xi1>, vector<3x1xi32>
// CHECK:         return {{%.+}} : vector<3x2xi32>
func.func @select_vector_elementwise(%a : vector<3xi64>, %b : vector<3xi64>, %c : vector<3xi1>) -> vector<3xi64> {
    %r = arith.select %c, %a, %b : vector<3xi1>, vector<3xi64>
    return %r : vector<3xi64>
}

// CHECK-LABEL: func.func @muli_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]  = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]] = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]  = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]] = vector.extract [[ARG1]][1] : vector<2xi32>
//
// CHECK-DAG:     [[RESLOW:%.+]], [[HI0:%.+]] = arith.mului_extended [[LOW0]], [[LOW1]] : i32
// CHECK-DAG:     [[HI1:%.+]]                 = arith.muli [[LOW0]], [[HIGH1]] : i32
// CHECK-DAG:     [[HI2:%.+]]                 = arith.muli [[HIGH0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RESHI1:%.+]]              = arith.addi [[HI0]], [[HI1]] : i32
// CHECK-NEXT:    [[RESHI2:%.+]]              = arith.addi [[RESHI1]], [[HI2]] : i32
//
// CHECK-NEXT:    [[VZ:%.+]]   = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]] = vector.insert [[RESLOW]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]] = vector.insert [[RESHI2]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @muli_scalar(%a : i64, %b : i64) -> i64 {
    %m = arith.muli %a, %b : i64
    return %m : i64
}

// CHECK-LABEL: func.func @muli_vector
// CHECK-SAME:    ({{%.+}}: vector<3x2xi32>, {{%.+}}: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK-DAG:     arith.mului_extended
// CHECK-DAG:     arith.muli
// CHECK-DAG:     arith.muli
// CHECK-NEXT:    arith.addi
// CHECK-NEXT:    arith.addi
// CHECK:       return {{%.+}} : vector<3x2xi32>
func.func @muli_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %m = arith.muli %a, %b : vector<3xi64>
    return %m : vector<3xi64>
}

// CHECK-LABEL: func.func @shli_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]     = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]    = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]     = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[CST0:%.+]]     = arith.constant 0 : i32
// CHECK-NEXT:    [[CST32:%.+]]    = arith.constant 32 : i32
// CHECK-NEXT:    [[OOB:%.+]]      = arith.cmpi uge, [[LOW1]], [[CST32]] : i32
// CHECK-NEXT:    [[SHLOW0:%.+]]   = arith.shli [[LOW0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RES0:%.+]]     = arith.select [[OOB]], [[CST0]], [[SHLOW0]] : i32
// CHECK-NEXT:    [[SHAMT:%.+]]    = arith.select [[OOB]], [[CST32]], [[LOW1]] : i32
// CHECK-NEXT:    [[RSHAMT:%.+]]   = arith.subi [[CST32]], [[SHAMT]] : i32
// CHECK-NEXT:    [[SHRHIGH0:%.+]] = arith.shrui [[LOW0]], [[RSHAMT]] : i32
// CHECK-NEXT:    [[LSHAMT:%.+]]   = arith.subi [[LOW1]], [[CST32]] : i32
// CHECK-NEXT:    [[SHLHIGH0:%.+]] = arith.shli [[LOW0]], [[LSHAMT]] : i32
// CHECK-NEXT:    [[SHLHIGH1:%.+]] = arith.shli [[HIGH0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RES1HIGH:%.+]] = arith.select [[OOB]], [[CST0]], [[SHLHIGH1]] : i32
// CHECK-NEXT:    [[RES1LOW:%.+]]  = arith.select [[OOB]], [[SHLHIGH0]], [[SHRHIGH0]] : i32
// CHECK-NEXT:    [[RES1:%.+]]     = arith.ori [[RES1LOW]], [[RES1HIGH]] : i32
// CHECK-NEXT:    [[VZ:%.+]]       = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]]     = vector.insert [[RES0]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]     = vector.insert [[RES1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @shli_scalar(%a : i64, %b : i64) -> i64 {
    %c = arith.shli %a, %b : i64
    return %c : i64
}

// CHECK-LABEL: func.func @shli_vector
// CHECK-SAME:    ({{%.+}}: vector<3x2xi32>, {{%.+}}: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:       return {{%.+}} : vector<3x2xi32>
func.func @shli_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %m = arith.shli %a, %b : vector<3xi64>
    return %m : vector<3xi64>
}

// CHECK-LABEL: func.func @shrui_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]     = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]    = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]     = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[CST0:%.+]]     = arith.constant 0 : i32
// CHECK-NEXT:    [[CST32:%.+]]    = arith.constant 32 : i32
// CHECK-DAG:     [[OOB:%.+]]      = arith.cmpi uge, [[LOW1]], [[CST32]] : i32
// CHECK-DAG:     [[SHLOW0:%.+]]   = arith.shrui [[LOW0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RES0LOW:%.+]]  = arith.select [[OOB]], [[CST0]], [[SHLOW0]] : i32
// CHECK-NEXT:    [[SHRHIGH0:%.+]] = arith.shrui [[HIGH0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RESLOW1:%.+]]  = arith.select [[OOB]], [[CST0]], [[SHRHIGH0]] : i32
// CHECK-NEXT:    [[SHAMT:%.+]]    = arith.select [[OOB]], [[CST32]], [[LOW1]] : i32
// CHECK-NEXT:    [[LSHAMT:%.+]]   = arith.subi [[CST32]], [[SHAMT]] : i32
// CHECK-NEXT:    [[SHLHIGH0:%.+]] = arith.shli [[HIGH0]], [[LSHAMT]] : i32
// CHECK-NEXT:    [[RSHAMT:%.+]]   = arith.subi [[LOW1]], [[CST32]] : i32
// CHECK-NEXT:    [[SHRHIGH0:%.+]] = arith.shrui [[HIGH0]], [[RSHAMT]] : i32
// CHECK-NEXT:    [[RES0HIGH:%.+]] = arith.select [[OOB]], [[SHRHIGH0]], [[SHLHIGH0]] : i32
// CHECK-NEXT:    [[RES0:%.+]]     = arith.ori [[RES0LOW]], [[RES0HIGH]] : i32
// CHECK-NEXT:    [[VZ:%.+]]       = arith.constant dense<0> : vector<2xi32>
// CHECK-NEXT:    [[INS0:%.+]]     = vector.insert [[RES0]], [[VZ]] [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]     = vector.insert [[RESLOW1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @shrui_scalar(%a : i64, %b : i64) -> i64 {
    %c = arith.shrui %a, %b : i64
    return %c : i64
}

// CHECK-LABEL: func.func @shrui_scalar_cst_2
// CHECK-SAME:    ({{%.+}}: vector<2xi32>) -> vector<2xi32>
// CHECK:       return {{%.+}} : vector<2xi32>
func.func @shrui_scalar_cst_2(%a : i64) -> i64 {
    %b = arith.constant 2 : i64
    %c = arith.shrui %a, %b : i64
    return %c : i64
}

// CHECK-LABEL: func.func @shrui_scalar_cst_36
// CHECK-SAME:    ({{%.+}}: vector<2xi32>) -> vector<2xi32>
// CHECK:       return {{%.+}} : vector<2xi32>
func.func @shrui_scalar_cst_36(%a : i64) -> i64 {
    %b = arith.constant 36 : i64
    %c = arith.shrui %a, %b : i64
    return %c : i64
}

// CHECK-LABEL: func.func @shrui_vector
// CHECK-SAME:    ({{%.+}}: vector<3x2xi32>, {{%.+}}: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:       return {{%.+}} : vector<3x2xi32>
func.func @shrui_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %m = arith.shrui %a, %b : vector<3xi64>
    return %m : vector<3xi64>
}

// CHECK-LABEL: func.func @shrsi_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]    = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]     = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[CST0:%.+]]     = arith.constant 0 : i32
// CHECK-NEXT:    [[NEG:%.+]]      = arith.cmpi slt, [[HIGH0]], [[CST0]] : i32
// CHECK-NEXT:    [[NEGEXT:%.+]]   = arith.extsi [[NEG]] : i1 to i32
// CHECK:         [[CST64:%.+]]    = arith.constant 64 : i32
// CHECK-NEXT:    [[SIGNBITS:%.+]] = arith.subi [[CST64]], [[LOW1]] : i32
// CHECK:         arith.shli
// CHECK:         arith.shrui
// CHECK:         arith.shli
// CHECK:         arith.shli
// CHECK:         arith.shrui
// CHECK:         arith.shrui
// CHECK:         arith.shli
// CHECK:         arith.shrui
// CHECK:         return {{%.+}} : vector<2xi32>
func.func @shrsi_scalar(%a : i64, %b : i64) -> i64 {
    %c = arith.shrsi %a, %b : i64
    return %c : i64
}

// CHECK-LABEL: func.func @shrsi_vector
// CHECK-SAME:    ({{%.+}}: vector<3x2xi32>, {{%.+}}: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         return {{%.+}} : vector<3x2xi32>
func.func @shrsi_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %m = arith.shrsi %a, %b : vector<3xi64>
    return %m : vector<3xi64>
}

// CHECK-LABEL: func @andi_scalar_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK-NEXT:    [[RES0:%.+]]   = arith.andi [[LOW0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RES1:%.+]]   = arith.andi [[HIGH0]], [[HIGH1]] : i32
// CHECK:         [[INS0:%.+]]   = vector.insert [[RES0]], {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert [[RES1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @andi_scalar_a_b(%a : i64, %b : i64) -> i64 {
    %x = arith.andi %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @andi_vector_a_b
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         {{%.+}} = arith.andi {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    {{%.+}} = arith.andi {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @andi_vector_a_b(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.andi %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func @ori_scalar_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK-NEXT:    [[RES0:%.+]]   = arith.ori [[LOW0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RES1:%.+]]   = arith.ori [[HIGH0]], [[HIGH1]] : i32
// CHECK:         [[INS0:%.+]]   = vector.insert [[RES0]], {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert [[RES1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @ori_scalar_a_b(%a : i64, %b : i64) -> i64 {
    %x = arith.ori %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @ori_vector_a_b
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         {{%.+}} = arith.ori {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    {{%.+}} = arith.ori {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @ori_vector_a_b(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.ori %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func @xori_scalar_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK-NEXT:    [[RES0:%.+]]   = arith.xori [[LOW0]], [[LOW1]] : i32
// CHECK-NEXT:    [[RES1:%.+]]   = arith.xori [[HIGH0]], [[HIGH1]] : i32
// CHECK:         [[INS0:%.+]]   = vector.insert [[RES0]], {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert [[RES1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @xori_scalar_a_b(%a : i64, %b : i64) -> i64 {
    %x = arith.xori %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @xori_vector_a_b
// CHECK-SAME:   ([[ARG0:%.+]]: vector<3x2xi32>, [[ARG1:%.+]]: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         {{%.+}} = arith.xori {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK-NEXT:    {{%.+}} = arith.xori {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         return {{.+}} : vector<3x2xi32>
func.func @xori_vector_a_b(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %x = arith.xori %a, %b : vector<3xi64>
    return %x : vector<3xi64>
}

// CHECK-LABEL: func @uitofp_i64_f64
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> f64
// CHECK-NEXT:    [[LOW:%.+]]    = vector.extract [[ARG]][0] : vector<2xi32>
// CHECK-NEXT:    [[HI:%.+]]     = vector.extract [[ARG]][1] : vector<2xi32>
// CHECK-NEXT:    [[CST0:%.+]]   = arith.constant 0 : i32
// CHECK-NEXT:    [[HIEQ0:%.+]]  = arith.cmpi eq, [[HI]], [[CST0]] : i32
// CHECK-NEXT:    [[LOWFP:%.+]]  = arith.uitofp [[LOW]] : i32 to f64
// CHECK-NEXT:    [[HIFP:%.+]]   = arith.uitofp [[HI]] : i32 to f64
// CHECK-NEXT:    [[POW:%.+]]    = arith.constant 0x41F0000000000000 : f64
// CHECK-NEXT:    [[RESHI:%.+]]  = arith.mulf [[HIFP]], [[POW]] : f64
// CHECK-NEXT:    [[RES:%.+]]    = arith.addf [[LOWFP]], [[RESHI]] : f64
// CHECK-NEXT:    [[SEL:%.+]]    = arith.select [[HIEQ0]], [[LOWFP]], [[RES]] : f64
// CHECK-NEXT:    return [[SEL]] : f64
func.func @uitofp_i64_f64(%a : i64) -> f64 {
    %r = arith.uitofp %a : i64 to f64
    return %r : f64
}

// CHECK-LABEL: func @uitofp_i64_f64_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3x2xi32>) -> vector<3xf64>
// CHECK-NEXT:    [[EXTLOW:%.+]] = vector.extract_strided_slice [[ARG]] {offsets = [0, 0], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[EXTHI:%.+]]  = vector.extract_strided_slice [[ARG]] {offsets = [0, 1], sizes = [3, 1], strides = [1, 1]} : vector<3x2xi32> to vector<3x1xi32>
// CHECK-NEXT:    [[LOW:%.+]]    = vector.shape_cast [[EXTLOW]] : vector<3x1xi32> to vector<3xi32>
// CHECK-NEXT:    [[HI:%.+]]     = vector.shape_cast [[EXTHI]] : vector<3x1xi32> to vector<3xi32>
// CHECK-NEXT:    [[CST0:%.+]]   = arith.constant dense<0> : vector<3xi32>
// CHECK-NEXT:    [[HIEQ0:%.+]]  = arith.cmpi eq, [[HI]], [[CST0]] : vector<3xi32>
// CHECK-NEXT:    [[LOWFP:%.+]]  = arith.uitofp [[LOW]] : vector<3xi32> to vector<3xf64>
// CHECK-NEXT:    [[HIFP:%.+]]   = arith.uitofp [[HI]] : vector<3xi32> to vector<3xf64>
// CHECK-NEXT:    [[POW:%.+]]    = arith.constant dense<0x41F0000000000000> : vector<3xf64>
// CHECK-NEXT:    [[RESHI:%.+]]  = arith.mulf [[HIFP]], [[POW]] : vector<3xf64>
// CHECK-NEXT:    [[RES:%.+]]    = arith.addf [[LOWFP]], [[RESHI]] : vector<3xf64>
// CHECK-NEXT:    [[SEL:%.+]]    = arith.select [[HIEQ0]], [[LOWFP]], [[RES]] : vector<3xi1>, vector<3xf64>
// CHECK-NEXT:    return [[SEL]] : vector<3xf64>
func.func @uitofp_i64_f64_vector(%a : vector<3xi64>) -> vector<3xf64> {
    %r = arith.uitofp %a : vector<3xi64> to vector<3xf64>
    return %r : vector<3xf64>
}

// CHECK-LABEL: func @uitofp_i64_f16
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> f16
// CHECK-NEXT:    [[LOW:%.+]]   = vector.extract [[ARG]][0] : vector<2xi32>
// CHECK-NEXT:    [[HI:%.+]]    = vector.extract [[ARG]][1] : vector<2xi32>
// CHECK-NEXT:    [[CST0:%.+]]   = arith.constant 0 : i32
// CHECK-NEXT:    [[HIEQ0:%.+]]  = arith.cmpi eq, [[HI]], [[CST0]] : i32
// CHECK-NEXT:    [[LOWFP:%.+]]  = arith.uitofp [[LOW]] : i32 to f16
// CHECK-NEXT:    [[HIFP:%.+]]   = arith.uitofp [[HI]] : i32 to f16
// CHECK-NEXT:    [[POW:%.+]]    = arith.constant 0x7C00 : f16
// CHECK-NEXT:    [[RESHI:%.+]]  = arith.mulf [[HIFP]], [[POW]] : f16
// CHECK-NEXT:    [[RES:%.+]]    = arith.addf [[LOWFP]], [[RESHI]] : f16
// CHECK-NEXT:    [[SEL:%.+]]    = arith.select [[HIEQ0]], [[LOWFP]], [[RES]] : f16
// CHECK-NEXT:    return [[SEL]] : f16
func.func @uitofp_i64_f16(%a : i64) -> f16 {
    %r = arith.uitofp %a : i64 to f16
    return %r : f16
}

// CHECK-LABEL: func @sitofp_i64_f64
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> f64
// CHECK:         [[VONES:%.+]]  = arith.constant dense<-1> : vector<2xi32>
// CHECK:         [[ONES1:%.+]]  = vector.extract [[VONES]][0] : vector<2xi32>
// CHECK-NEXT:    [[ONES2:%.+]]  = vector.extract [[VONES]][1] : vector<2xi32>
// CHECK:                          arith.xori {{%.+}}, [[ONES1]] : i32
// CHECK-NEXT:                     arith.xori {{%.+}}, [[ONES2]] : i32
// CHECK:         [[CST0:%.+]]   = arith.constant 0 : i32
// CHECK:         [[HIEQ0:%.+]]  = arith.cmpi eq, [[HI:%.+]], [[CST0]] : i32
// CHECK-NEXT:    [[LOWFP:%.+]]  = arith.uitofp [[LOW:%.+]] : i32 to f64
// CHECK-NEXT:    [[HIFP:%.+]]   = arith.uitofp [[HI]] : i32 to f64
// CHECK-NEXT:    [[POW:%.+]]    = arith.constant 0x41F0000000000000 : f64
// CHECK-NEXT:    [[RESHI:%.+]]  = arith.mulf [[HIFP]], [[POW]] : f64
// CHECK-NEXT:    [[RES:%.+]]    = arith.addf [[LOWFP]], [[RESHI]] : f64
// CHECK-NEXT:    [[SEL:%.+]]    = arith.select [[HIEQ0]], [[LOWFP]], [[RES]] : f64
// CHECK-NEXT:    [[NEG:%.+]]    = arith.negf [[SEL]] : f64
// CHECK-NEXT:    [[FINAL:%.+]]  = arith.select %{{.+}}, [[NEG]], [[SEL]] : f64
// CHECK-NEXT:    return [[FINAL]] : f64
func.func @sitofp_i64_f64(%a : i64) -> f64 {
    %r = arith.sitofp %a : i64 to f64
    return %r : f64
}

// CHECK-LABEL: func @sitofp_i64_f64_vector
// CHECK-SAME:    ([[ARG:%.+]]: vector<3x2xi32>) -> vector<3xf64>
// CHECK:         [[VONES:%.+]]  = arith.constant dense<-1> : vector<3x2xi32>
// CHECK:                          arith.xori
// CHECK-NEXT:                     arith.xori
// CHECK:         [[HIEQ0:%.+]]  = arith.cmpi eq, [[HI:%.+]], [[CST0:%.+]] : vector<3xi32>
// CHECK-NEXT:    [[LOWFP:%.+]]  = arith.uitofp [[LOW:%.+]] : vector<3xi32> to vector<3xf64>
// CHECK-NEXT:    [[HIFP:%.+]]   = arith.uitofp [[HI:%.+]] : vector<3xi32> to vector<3xf64>
// CHECK-NEXT:    [[POW:%.+]]    = arith.constant dense<0x41F0000000000000> : vector<3xf64>
// CHECK-NEXT:    [[RESHI:%.+]]  = arith.mulf [[HIFP]], [[POW]] : vector<3xf64>
// CHECK-NEXT:    [[RES:%.+]]    = arith.addf [[LOWFP]], [[RESHI]] : vector<3xf64>
// CHECK-NEXT:    [[SEL:%.+]]    = arith.select [[HIEQ0]], [[LOWFP]], [[RES]] : vector<3xi1>, vector<3xf64>
// CHECK-NEXT:    [[NEG:%.+]]    = arith.negf [[SEL]] : vector<3xf64>
// CHECK-NEXT:    [[FINAL:%.+]]  = arith.select %{{.+}}, [[NEG]], [[SEL]] : vector<3xi1>, vector<3xf64>
// CHECK-NEXT:    return [[FINAL]] : vector<3xf64>
func.func @sitofp_i64_f64_vector(%a : vector<3xi64>) -> vector<3xf64> {
    %r = arith.sitofp %a : vector<3xi64> to vector<3xf64>
    return %r : vector<3xf64>
}
