// RUN: mlir-opt %s -convert-vector-to-llvm="enable-vcix" -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s

// -----
// CHECK-LABEL:   llvm.func @unary_ro_e8mf8(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 0 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8mf8(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8mf8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e8mf4(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 1 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8mf4(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8mf4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e8mf2(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 2 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8mf2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8mf2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e8m1(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 3 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8m1(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e8m2(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 4 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8m2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e8m4(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 5 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8m4(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e8m8(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 6 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e8m8(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
// CHECK-LABEL:   llvm.func @unary_e8mf8(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[1]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[1]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[1]xi8>
// CHECK:         }
func.func @unary_e8mf8(%rvl: ui32) -> vector<[1] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i8>
  return %0 : vector<[1] x i8>
}

// CHECK-LABEL:   llvm.func @unary_e8mf4(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[2]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[2]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[2]xi8>
// CHECK:         }
func.func @unary_e8mf4(%rvl: ui32) -> vector<[2] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i8>
  return %0 : vector<[2] x i8>
}

// CHECK-LABEL:   llvm.func @unary_e8mf2(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[4]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[4]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[4]xi8>
// CHECK:         }
func.func @unary_e8mf2(%rvl: ui32) -> vector<[4] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i8>
  return %0 : vector<[4] x i8>
}

// CHECK-LABEL:   llvm.func @unary_e8m1(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32) -> vector<[8]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[8]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[8]xi8>
// CHECK:         }
func.func @unary_e8m1(%rvl: ui32) -> vector<[8] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i8>
  return %0 : vector<[8] x i8>
}

// CHECK-LABEL:   llvm.func @unary_e8m2(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32) -> vector<[16]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[16]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[16]xi8>
// CHECK:         }
func.func @unary_e8m2(%rvl: ui32) -> vector<[16] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[16] x i8>
  return %0 : vector<[16] x i8>
}

// CHECK-LABEL:   llvm.func @unary_e8m4(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32) -> vector<[32]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[32]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[32]xi8>
// CHECK:         }
func.func @unary_e8m4(%rvl: ui32) -> vector<[32] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[32] x i8>
  return %0 : vector<[32] x i8>
}

// CHECK-LABEL:   llvm.func @unary_e8m8(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32) -> vector<[64]xi8> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[64]xi8>
// CHECK:           llvm.return %[[VAL_2]] : vector<[64]xi8>
// CHECK:         }
func.func @unary_e8m8(%rvl: ui32) -> vector<[64] x i8> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[64] x i8>
  return %0 : vector<[64] x i8>
}

// -----
// CHECK-LABEL:   llvm.func @unary_ro_e16mf4(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 7 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e16mf4(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16mf4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e16mf2(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 8 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e16mf2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16mf2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e16m1(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 9 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e16m1(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e16m2(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 10 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e16m2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e16m4(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 11 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e16m4(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e16m8(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 12 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e16m8(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
// CHECK-LABEL:   llvm.func @unary_e16mf4(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32) -> vector<[1]xi16> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[1]xi16>
// CHECK:           llvm.return %[[VAL_2]] : vector<[1]xi16>
// CHECK:         }
func.func @unary_e16mf4(%rvl: ui32) -> vector<[1] x i16> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i16>
  return %0 : vector<[1] x i16>
}

// CHECK-LABEL:   llvm.func @unary_e16mf2(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32) -> vector<[2]xi16> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[2]xi16>
// CHECK:           llvm.return %[[VAL_2]] : vector<[2]xi16>
// CHECK:         }
func.func @unary_e16mf2(%rvl: ui32) -> vector<[2] x i16> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i16>
  return %0 : vector<[2] x i16>
}

// CHECK-LABEL:   llvm.func @unary_e16m1(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[4]xi16> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[4]xi16>
// CHECK:           llvm.return %[[VAL_2]] : vector<[4]xi16>
// CHECK:         }
func.func @unary_e16m1(%rvl: ui32) -> vector<[4] x i16> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i16>
  return %0 : vector<[4] x i16>
}

// CHECK-LABEL:   llvm.func @unary_e16m2(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[8]xi16> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[8]xi16>
// CHECK:           llvm.return %[[VAL_2]] : vector<[8]xi16>
// CHECK:         }
func.func @unary_e16m2(%rvl: ui32) -> vector<[8] x i16> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i16>
  return %0 : vector<[8] x i16>
}

// CHECK-LABEL:   llvm.func @unary_e16m4(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[16]xi16> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[16]xi16>
// CHECK:           llvm.return %[[VAL_2]] : vector<[16]xi16>
// CHECK:         }
func.func @unary_e16m4(%rvl: ui32) -> vector<[16] x i16> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[16] x i16>
  return %0 : vector<[16] x i16>
}

// CHECK-LABEL:   llvm.func @unary_e16m8(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[32]xi16> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[32]xi16>
// CHECK:           llvm.return %[[VAL_2]] : vector<[32]xi16>
// CHECK:         }
func.func @unary_e16m8(%rvl: ui32) -> vector<[32] x i16> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[32] x i16>
  return %0 : vector<[32] x i16>
}

// -----
// CHECK-LABEL:   llvm.func @unary_ro_e32mf2(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 13 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e32mf2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32mf2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e32m1(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 14 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e32m1(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e32m2(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 15 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e32m2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e32m4(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 16 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e32m4(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e32m8(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 17 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e32m8(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
// CHECK-LABEL:   llvm.func @unary_e32mf2(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32) -> vector<[1]xi32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[1]xi32>
// CHECK:           llvm.return %[[VAL_2]] : vector<[1]xi32>
// CHECK:         }
func.func @unary_e32mf2(%rvl: ui32) -> vector<[1] x i32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i32>
  return %0 : vector<[1] x i32>
}

// CHECK-LABEL:   llvm.func @unary_e32m1(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[2]xi32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[2]xi32>
// CHECK:           llvm.return %[[VAL_2]] : vector<[2]xi32>
// CHECK:         }
func.func @unary_e32m1(%rvl: ui32) -> vector<[2] x i32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i32>
  return %0 : vector<[2] x i32>
}

// CHECK-LABEL:   llvm.func @unary_e32m2(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[4]xi32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[4]xi32>
// CHECK:           llvm.return %[[VAL_2]] : vector<[4]xi32>
// CHECK:         }
func.func @unary_e32m2(%rvl: ui32) -> vector<[4] x i32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i32>
  return %0 : vector<[4] x i32>
}

// CHECK-LABEL:   llvm.func @unary_e32m4(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[8]xi32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[8]xi32>
// CHECK:           llvm.return %[[VAL_2]] : vector<[8]xi32>
// CHECK:         }
func.func @unary_e32m4(%rvl: ui32) -> vector<[8] x i32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i32>
  return %0 : vector<[8] x i32>
}

// CHECK-LABEL:   llvm.func @unary_e32m8(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[16]xi32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[16]xi32>
// CHECK:           llvm.return %[[VAL_2]] : vector<[16]xi32>
// CHECK:         }
func.func @unary_e32m8(%rvl: ui32) -> vector<[16] x i32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[16] x i32>
  return %0 : vector<[16] x i32>
}

// -----
// CHECK-LABEL:   llvm.func @unary_ro_e64m1(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 18 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e64m1(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e64m2(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 19 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e64m2(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e64m4(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 20 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e64m4(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// CHECK-LABEL:   llvm.func @unary_ro_e64m8(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           "vcix.intrin.unary.ro"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32, rs2 = 31 : i32, sew_lmul = 21 : i32}> : (i32, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @unary_ro_e64m8(%rvl: ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
// CHECK-LABEL:   llvm.func @unary_e64m1(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[1]xi64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[1]xi64>
// CHECK:           llvm.return %[[VAL_2]] : vector<[1]xi64>
// CHECK:         }
func.func @unary_e64m1(%rvl: ui32) -> vector<[1] x i64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i64>
  return %0 : vector<[1] x i64>
}

// CHECK-LABEL:   llvm.func @unary_e64m2(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[2]xi64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[2]xi64>
// CHECK:           llvm.return %[[VAL_2]] : vector<[2]xi64>
// CHECK:         }
func.func @unary_e64m2(%rvl: ui32) -> vector<[2] x i64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i64>
  return %0 : vector<[2] x i64>
}

// CHECK-LABEL:   llvm.func @unary_e64m4(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[4]xi64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[4]xi64>
// CHECK:           llvm.return %[[VAL_2]] : vector<[4]xi64>
// CHECK:         }
func.func @unary_e64m4(%rvl: ui32) -> vector<[4] x i64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i64>
  return %0 : vector<[4] x i64>
}

// CHECK-LABEL:   llvm.func @unary_e64m8(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> vector<[8]xi64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.unary"(%[[VAL_1]], %[[VAL_0]]) <{opcode = 3 : i32, rs2 = 31 : i32}> : (i32, i32) -> vector<[8]xi64>
// CHECK:           llvm.return %[[VAL_2]] : vector<[8]xi64>
// CHECK:         }
func.func @unary_e64m8(%rvl: ui32) -> vector<[8] x i64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i64>
  return %0 : vector<[8] x i64>
}

// -----
// CHECK-LABEL:   llvm.func @binary_vv_ro(
// CHECK-SAME:                            %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32, rd = 30 : i32}> : (vector<[4]xf32>, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @binary_vv_ro(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.binary.ro %op1, %op2, %rvl { opcode = 3 : i2, rd = 30 : i5 } : (vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @binary_vv(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.binary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (vector<[4]xf32>, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<[4]xf32>
// CHECK:         }
func.func @binary_vv(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// CHECK-LABEL:   llvm.func @binary_xv_ro(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32, rd = 30 : i32}> : (i32, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @binary_xv_ro(%op1: i32, %op2 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.binary.ro %op1, %op2, %rvl { opcode = 3 : i2, rd = 30 : i5 } : (i32, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @binary_xv(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.binary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<[4]xf32>
// CHECK:         }
func.func @binary_xv(%op1: i32, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// CHECK-LABEL:   llvm.func @binary_fv_ro(
// CHECK-SAME:                            %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 1 : i32, rd = 30 : i32}> : (f32, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @binary_fv_ro(%op1: f32, %op2 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.binary.ro %op1, %op2, %rvl { opcode = 1 : i1, rd = 30 : i5 } : (f32, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @binary_fv(
// CHECK-SAME:                         %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.binary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 1 : i32}> : (f32, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<[4]xf32>
// CHECK:         }
func.func @binary_fv(%op1: f32, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// CHECK-LABEL:   llvm.func @binary_iv_ro(
// CHECK-SAME:                            %[[VAL_0:.*]]: vector<[4]xf32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i5 to i32
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32, rd = 30 : i32}> : (i32, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @binary_iv_ro(%op2 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  vcix.binary.ro %const, %op2, %rvl { opcode = 3 : i2, rd = 30 : i5 } : (i5, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @binary_iv(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<[4]xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i5 to i32
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.binary"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_4]] : vector<[4]xf32>
// CHECK:         }
func.func @binary_iv(%op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  %0 = vcix.binary %const, %op2, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
// CHECK-LABEL:   llvm.func @ternary_vvv_ro(
// CHECK-SAME:                              %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf32>, %[[VAL_3:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @ternary_vvv_ro(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @ternary_vvv(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf32>, %[[VAL_3:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_4]] : vector<[4]xf32>
// CHECK:         }
func.func @ternary_vvv(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// CHECK-LABEL:   llvm.func @ternary_xvv_ro(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf32>, %[[VAL_3:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @ternary_xvv_ro(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @ternary_xvv(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf32>, %[[VAL_3:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_4]] : vector<[4]xf32>
// CHECK:         }
func.func @ternary_xvv(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// CHECK-LABEL:   llvm.func @ternary_fvv_ro(
// CHECK-SAME:                              %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf32>, %[[VAL_3:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 1 : i32}> : (f32, vector<[4]xf32>, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @ternary_fvv_ro(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.ternary.ro %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @ternary_fvv(
// CHECK-SAME:                           %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf32>, %[[VAL_3:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 1 : i32}> : (f32, vector<[4]xf32>, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_4]] : vector<[4]xf32>
// CHECK:         }
func.func @ternary_fvv(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// CHECK-LABEL:   llvm.func @ternary_ivv_ro(
// CHECK-SAME:                              %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_4:.*]] = llvm.zext %[[VAL_3]] : i5 to i32
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf32>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @ternary_ivv_ro(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  vcix.ternary.ro %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @ternary_ivv(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: i32) -> vector<[4]xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_4:.*]] = llvm.zext %[[VAL_3]] : i5 to i32
// CHECK:           %[[VAL_5:.*]] = "vcix.intrin.ternary"(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf32>, i32) -> vector<[4]xf32>
// CHECK:           llvm.return %[[VAL_5]] : vector<[4]xf32>
// CHECK:         }
func.func @ternary_ivv(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  %0 = vcix.ternary %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
// CHECK-LABEL:   llvm.func @wide_ternary_vvw_ro(
// CHECK-SAME:                                   %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf64>, %[[VAL_3:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf64>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @wide_ternary_vvw_ro(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.wide.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @wide_ternary_vvw(
// CHECK-SAME:                                %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf64>, %[[VAL_3:.*]]: i32) -> vector<[4]xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf64>, i32) -> vector<[4]xf64>
// CHECK:           llvm.return %[[VAL_4]] : vector<[4]xf64>
// CHECK:         }
func.func @wide_ternary_vvw(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %0: vector<[4] x f64>
}

// CHECK-LABEL:   llvm.func @wide_ternary_xvw_ro(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf64>, %[[VAL_3:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf64>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @wide_ternary_xvw_ro(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.wide.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @wide_ternary_xvw(
// CHECK-SAME:                                %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf64>, %[[VAL_3:.*]]: i32) -> vector<[4]xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf64>, i32) -> vector<[4]xf64>
// CHECK:           llvm.return %[[VAL_4]] : vector<[4]xf64>
// CHECK:         }
func.func @wide_ternary_xvw(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %0 : vector<[4] x f64>
}

// CHECK-LABEL:   llvm.func @wide_ternary_fvw_ro(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf64>, %[[VAL_3:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 1 : i32}> : (f32, vector<[4]xf32>, vector<[4]xf64>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @wide_ternary_fvw_ro(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.wide.ternary.ro %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @wide_ternary_fvw(
// CHECK-SAME:                                %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<[4]xf32>, %[[VAL_2:.*]]: vector<[4]xf64>, %[[VAL_3:.*]]: i32) -> vector<[4]xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{opcode = 1 : i32}> : (f32, vector<[4]xf32>, vector<[4]xf64>, i32) -> vector<[4]xf64>
// CHECK:           llvm.return %[[VAL_2]] : vector<[4]xf64>
// CHECK:         }
func.func @wide_ternary_fvw(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %op3 : vector<[4] x f64>
}

// CHECK-LABEL:   llvm.func @wide_ternary_ivw_ro(
// CHECK-SAME:                                   %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf64>, %[[VAL_2:.*]]: i32) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_4:.*]] = llvm.zext %[[VAL_3]] : i5 to i32
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf64>, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @wide_ternary_ivw_ro(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  vcix.wide.ternary.ro %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

// CHECK-LABEL:   llvm.func @wide_ternary_ivv(
// CHECK-SAME:                                %[[VAL_0:.*]]: vector<[4]xf32>, %[[VAL_1:.*]]: vector<[4]xf64>, %[[VAL_2:.*]]: i32) -> vector<[4]xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_4:.*]] = llvm.zext %[[VAL_3]] : i5 to i32
// CHECK:           %[[VAL_5:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<[4]xf32>, vector<[4]xf64>, i32) -> vector<[4]xf64>
// CHECK:           llvm.return %[[VAL_1]] : vector<[4]xf64>
// CHECK:         }
func.func @wide_ternary_ivv(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  %0 = vcix.wide.ternary %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %op3 : vector<[4] x f64>
}

// -----
// CHECK-LABEL:   llvm.func @fixed_binary_vv_ro(
// CHECK-SAME:                                  %[[VAL_0:.*]]: vector<4xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32, rd = 30 : i32}> : (vector<4xf32>, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_binary_vv_ro(%op1: vector<4 x f32>, %op2 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.binary.ro %op1, %op2 { opcode = 3 : i2, rd = 30 : i5 } : (vector<4 x f32>, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_binary_vv(
// CHECK-SAME:                               %[[VAL_0:.*]]: vector<4xf32>,
// CHECK-SAME:                               %[[VAL_1:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.binary"(%[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_2]] : vector<4xf32>
// CHECK:         }
func.func @fixed_binary_vv(%op1: vector<4 x f32>, %op2 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.binary %op1, %op2 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// CHECK-LABEL:   llvm.func @fixed_binary_xv_ro(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                  %[[VAL_1:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32, rd = 30 : i32}> : (i32, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_binary_xv_ro(%op1: i32, %op2 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.binary.ro %op1, %op2 { opcode = 3 : i2, rd = 30 : i5 } : (i32, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_binary_xv(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32,
// CHECK-SAME:                               %[[VAL_1:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.binary"(%[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_2]] : vector<4xf32>
// CHECK:         }
func.func @fixed_binary_xv(%op1: i32, %op2 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.binary %op1, %op2 { opcode = 3 : i2 } : (i32, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// CHECK-LABEL:   llvm.func @fixed_binary_fv_ro(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                  %[[VAL_1:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_0]], %[[VAL_1]]) <{opcode = 1 : i32, rd = 30 : i32}> : (f32, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_binary_fv_ro(%op1: f32, %op2 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.binary.ro %op1, %op2 { opcode = 1 : i1, rd = 30 : i5 } : (f32, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_binary_fv(
// CHECK-SAME:                               %[[VAL_0:.*]]: f32,
// CHECK-SAME:                               %[[VAL_1:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = "vcix.intrin.binary"(%[[VAL_0]], %[[VAL_1]]) <{opcode = 1 : i32}> : (f32, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_2]] : vector<4xf32>
// CHECK:         }
func.func @fixed_binary_fv(%op1: f32, %op2 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.binary %op1, %op2 { opcode = 1 : i1 } : (f32, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// CHECK-LABEL:   llvm.func @fixed_binary_iv_ro(
// CHECK-SAME:                                  %[[VAL_0:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_2:.*]] = llvm.zext %[[VAL_1]] : i5 to i32
// CHECK:           "vcix.intrin.binary.ro"(%[[VAL_2]], %[[VAL_0]]) <{opcode = 3 : i32, rd = 30 : i32}> : (i32, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_binary_iv_ro(%op2 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  vcix.binary.ro %const, %op2 { opcode = 3 : i2, rd = 30 : i5 } : (i5, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_binary_iv(
// CHECK-SAME:                               %[[VAL_0:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_2:.*]] = llvm.zext %[[VAL_1]] : i5 to i32
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.binary"(%[[VAL_2]], %[[VAL_0]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xf32>
// CHECK:         }
func.func @fixed_binary_iv(%op2 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  %0 = vcix.binary %const, %op2 { opcode = 3 : i2 } : (i5, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// -----
// CHECK-LABEL:   llvm.func @fixed_ternary_vvv_ro(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_ternary_vvv_ro(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_ternary_vvv(
// CHECK-SAME:                                 %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xf32>
// CHECK:         }
func.func @fixed_ternary_vvv(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// CHECK-LABEL:   llvm.func @fixed_ternary_xvv_ro(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_ternary_xvv_ro(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_ternary_xvv(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xf32>
// CHECK:         }
func.func @fixed_ternary_xvv(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// CHECK-LABEL:   llvm.func @fixed_ternary_fvv_ro(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 1 : i32}> : (f32, vector<4xf32>, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_ternary_fvv_ro(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.ternary.ro %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_ternary_fvv(
// CHECK-SAME:                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 1 : i32}> : (f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xf32>
// CHECK:         }
func.func @fixed_ternary_fvv(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.ternary %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// CHECK-LABEL:   llvm.func @fixed_ternary_ivv_ro(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<4xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: vector<4xf32>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i5 to i32
// CHECK:           "vcix.intrin.ternary.ro"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf32>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_ternary_ivv_ro(%op2 : vector<4 x f32>, %op3 : vector<4 x f32>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  vcix.ternary.ro %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f32>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_ternary_ivv(
// CHECK-SAME:                                 %[[VAL_0:.*]]: vector<4xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: vector<4xf32>) -> vector<4xf32> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i5 to i32
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.ternary"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK:           llvm.return %[[VAL_4]] : vector<4xf32>
// CHECK:         }
func.func @fixed_ternary_ivv(%op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  %0 = vcix.ternary %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// -----
// CHECK-LABEL:   llvm.func @fixed_wide_ternary_vvw_ro(
// CHECK-SAME:                                         %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf64>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (vector<4xf32>, vector<4xf32>, vector<4xf64>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_wide_ternary_vvw_ro(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.wide.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f64>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_vvw(
// CHECK-SAME:                                      %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf64>) -> vector<4xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (vector<4xf32>, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xf64>
// CHECK:         }
func.func @fixed_wide_ternary_vvw(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.wide.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %0 : vector<4 x f64>
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_xvw_ro(
// CHECK-SAME:                                         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf64>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf64>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_wide_ternary_xvw_ro(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.wide.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f64>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_xvw(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf64>) -> vector<4xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xf64>
// CHECK:         }
func.func @fixed_wide_ternary_xvw(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.wide.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %0 : vector<4 x f64>
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_fvw_ro(
// CHECK-SAME:                                         %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf64>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 1 : i32}> : (f32, vector<4xf32>, vector<4xf64>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_wide_ternary_fvw_ro(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  vcix.wide.ternary.ro %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f64>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_fvw(
// CHECK-SAME:                                      %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: vector<4xf32>, %[[VAL_2:.*]]: vector<4xf64>) -> vector<4xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_3:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{opcode = 1 : i32}> : (f32, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
// CHECK:           llvm.return %[[VAL_2]] : vector<4xf64>
// CHECK:         }
func.func @fixed_wide_ternary_fvw(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %0 = vcix.wide.ternary %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %op3 : vector<4 x f64>
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_ivw_ro(
// CHECK-SAME:                                         %[[VAL_0:.*]]: vector<4xf32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: vector<4xf64>) attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i5 to i32
// CHECK:           "vcix.intrin.wide.ternary.ro"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf64>) -> ()
// CHECK:           llvm.return
// CHECK:         }
func.func @fixed_wide_ternary_ivw_ro(%op2 : vector<4 x f32>, %op3 : vector<4 x f64>) attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  vcix.wide.ternary.ro %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f64>)
  return
}

// CHECK-LABEL:   llvm.func @fixed_wide_ternary_ivv(
// CHECK-SAME:                                      %[[VAL_0:.*]]: vector<4xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: vector<4xf64>) -> vector<4xf64> attributes {vcix.target_features = "+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i5) : i5
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i5 to i32
// CHECK:           %[[VAL_4:.*]] = "vcix.intrin.wide.ternary"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{opcode = 3 : i32}> : (i32, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
// CHECK:           llvm.return %[[VAL_1]] : vector<4xf64>
// CHECK:         }
func.func @fixed_wide_ternary_ivv(%op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> attributes { vcix.target_features="+32bit,+v,+zfh,+xsfvcp,+zvl2048b,+zvfh,+v,+zmmul" } {
  %const = arith.constant 1 : i5
  %0 = vcix.wide.ternary %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %op3 : vector<4 x f64>
}
