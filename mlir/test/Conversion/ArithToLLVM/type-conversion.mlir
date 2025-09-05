// RUN: mlir-opt %s -test-llvm-legalize-patterns -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-llvm-legalize-patterns="allow-pattern-rollback=0" -split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @arith_select(
//  CHECK-SAME:     %[[arg0:.*]]: i1, %[[arg1:.*]]: i18, %[[arg2:.*]]: i18, %[[arg3:.*]]: i18, %[[arg4:.*]]: i18) -> !llvm.struct<(i18, i18)>
//       CHECK:   %[[select0:.*]] = llvm.select %[[arg0]], %[[arg1]], %[[arg3]] : i1, i18
//       CHECK:   %[[select1:.*]] = llvm.select %[[arg0]], %[[arg2]], %[[arg4]] : i1, i18
//       CHECK:   %[[i0:.*]] = llvm.mlir.poison : !llvm.struct<(i18, i18)>
//       CHECK:   %[[i1:.*]] = llvm.insertvalue %[[select0]], %[[i0]][0] : !llvm.struct<(i18, i18)>
//       CHECK:   %[[i2:.*]] = llvm.insertvalue %[[select1]], %[[i1]][1] : !llvm.struct<(i18, i18)>
//       CHECK:   llvm.return %[[i2]]
func.func @arith_select(%arg0: i1, %arg1: i17, %arg2: i17) -> (i17) {
  %0 = arith.select %arg0, %arg1, %arg2 : i17
  return %0 : i17
}
