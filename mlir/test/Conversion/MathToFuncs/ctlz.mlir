// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(convert-math-to-funcs{convert-ctlz})" | FileCheck %s
// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(convert-math-to-funcs{convert-ctlz=false})" | FileCheck --check-prefix=NOCVT %s

// Check a golden-path i32 conversion

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32
// CHECK-SAME:                       ) {
// CHECK:           %[[VAL_1:.*]] = call @__mlir_math_ctlz_i32(%[[VAL_0]]) : (i32) -> i32
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__mlir_math_ctlz_i32(
// CHECK-SAME:            %[[ARG:.*]]: i32
// CHECK-SAME:            ) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[C_32:.*]] = arith.constant 32 : i32
// CHECK:           %[[C_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[ARGCMP:.*]] = arith.cmpi eq, %[[ARG]], %[[C_0]] : i32
// CHECK:           %[[OUT:.*]] = scf.if %[[ARGCMP]] -> (i32) {
// CHECK:             scf.yield %[[C_32]] : i32
// CHECK:           } else {
// CHECK:             %[[C_1INDEX:.*]] = arith.constant 1 : index
// CHECK:             %[[C_1I32:.*]] = arith.constant 1 : i32
// CHECK:             %[[C_32INDEX:.*]] = arith.constant 32 : index
// CHECK:             %[[N:.*]] = arith.constant 0 : i32
// CHECK:             %[[FOR_RET:.*]]:2 = scf.for %[[I:.*]] = %[[C_1INDEX]] to %[[C_32INDEX]] step %[[C_1INDEX]]
// CHECK:                 iter_args(%[[ARG_ITER:.*]] = %[[ARG]], %[[N_ITER:.*]] = %[[N]]) -> (i32, i32) {
// CHECK:               %[[COND:.*]] = arith.cmpi slt, %[[ARG_ITER]], %[[C_0]] : i32
// CHECK:               %[[IF_RET:.*]]:2 = scf.if %[[COND]] -> (i32, i32) {
// CHECK:                 scf.yield %[[ARG_ITER]], %[[N_ITER]] : i32, i32
// CHECK:               } else {
// CHECK:                 %[[N_NEXT:.*]] = arith.addi %[[N_ITER]], %[[C_1I32]] : i32
// CHECK:                 %[[ARG_NEXT:.*]] = arith.shli %[[ARG_ITER]], %[[C_1I32]] : i32
// CHECK:                 scf.yield %[[ARG_NEXT]], %[[N_NEXT]] : i32, i32
// CHECK:               }
// CHECK:               scf.yield %[[IF_RET]]#0, %[[IF_RET]]#1 : i32, i32
// CHECK:             }
// CHECK:             scf.yield %[[FOR_RET]]#1 : i32
// CHECK:           }
// CHECK:           return %[[OUT]] : i32
// CHECK:         }
// NOCVT-NOT: __mlir_math_ctlz_i32
func.func @main(%arg0: i32) {
  %0 = math.ctlz %arg0 : i32
  func.return
}

// -----

// Check that i8 input is preserved

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                       %[[VAL_0:.*]]: i8
// CHECK-SAME:                       ) {
// CHECK:           %[[VAL_1:.*]] = call @__mlir_math_ctlz_i8(%[[VAL_0]]) : (i8) -> i8
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__mlir_math_ctlz_i8(
// CHECK-SAME:            %[[ARG:.*]]: i8
// CHECK-SAME:            ) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[C_8:.*]] = arith.constant 8 : i8
// CHECK:           %[[C_0:.*]] = arith.constant 0 : i8
// CHECK:           %[[ARGCMP:.*]] = arith.cmpi eq, %[[ARG]], %[[C_0]] : i8
// CHECK:           %[[OUT:.*]] = scf.if %[[ARGCMP]] -> (i8) {
// CHECK:             scf.yield %[[C_8]] : i8
// CHECK:           } else {
// CHECK:             %[[C_1INDEX:.*]] = arith.constant 1 : index
// CHECK:             %[[C_1I32:.*]] = arith.constant 1 : i8
// CHECK:             %[[C_8INDEX:.*]] = arith.constant 8 : index
// CHECK:             %[[N:.*]] = arith.constant 0 : i8
// CHECK:             %[[FOR_RET:.*]]:2 = scf.for %[[I:.*]] = %[[C_1INDEX]] to %[[C_8INDEX]] step %[[C_1INDEX]]
// CHECK:                 iter_args(%[[ARG_ITER:.*]] = %[[ARG]], %[[N_ITER:.*]] = %[[N]]) -> (i8, i8) {
// CHECK:               %[[COND:.*]] = arith.cmpi slt, %[[ARG_ITER]], %[[C_0]] : i8
// CHECK:               %[[IF_RET:.*]]:2 = scf.if %[[COND]] -> (i8, i8) {
// CHECK:                 scf.yield %[[ARG_ITER]], %[[N_ITER]] : i8, i8
// CHECK:               } else {
// CHECK:                 %[[N_NEXT:.*]] = arith.addi %[[N_ITER]], %[[C_1I32]] : i8
// CHECK:                 %[[ARG_NEXT:.*]] = arith.shli %[[ARG_ITER]], %[[C_1I32]] : i8
// CHECK:                 scf.yield %[[ARG_NEXT]], %[[N_NEXT]] : i8, i8
// CHECK:               }
// CHECK:               scf.yield %[[IF_RET]]#0, %[[IF_RET]]#1 : i8, i8
// CHECK:             }
// CHECK:             scf.yield %[[FOR_RET]]#1 : i8
// CHECK:           }
// CHECK:           return %[[OUT]] : i8
// CHECK:         }
// NOCVT-NOT: __mlir_math_ctlz_i32
func.func @main(%arg0: i8) {
  %0 = math.ctlz %arg0 : i8
  func.return
}

// -----

// Check that index is not converted

// CHECK-LABEL: func.func @ctlz_index
// CHECK:         math.ctlz
func.func @ctlz_index(%arg0: index) {
  %0 = math.ctlz %arg0 : index
  func.return
}
