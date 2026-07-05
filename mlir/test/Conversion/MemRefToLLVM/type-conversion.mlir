// RUN: mlir-opt %s -test-llvm-legalize-patterns -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-llvm-legalize-patterns="allow-pattern-rollback=0" -split-input-file | FileCheck %s

// Test the argument materializer for ranked MemRef types.

//   CHECK-LABEL: func @construct_ranked_memref_descriptor(
//         CHECK:   llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-COUNT-7:   llvm.insertvalue
//         CHECK:   builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<5x4xf32>
func.func @construct_ranked_memref_descriptor(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {is_legal} {
  %0 = "test.direct_replacement"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> (memref<5x4xf32>)
  "test.legal_op"(%0) : (memref<5x4xf32>) -> ()
  return
}

// -----

// The argument materializer for ranked MemRef types is called with incorrect
// input types. Make sure that the materializer is skipped and we do not
// generate invalid IR.

// CHECK-LABEL: func @invalid_ranked_memref_descriptor(
//       CHECK:   %[[cast:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i1 to memref<5x4xf32>
//       CHECK:   "test.legal_op"(%[[cast]])
func.func @invalid_ranked_memref_descriptor(%arg0: i1) attributes {is_legal} {
  %0 = "test.direct_replacement"(%arg0) : (i1) -> (memref<5x4xf32>)
  "test.legal_op"(%0) : (memref<5x4xf32>) -> ()
  return
}

// -----

// Test the argument materializer for unranked MemRef types.

//   CHECK-LABEL: func @construct_unranked_memref_descriptor(
//         CHECK:   llvm.mlir.poison : !llvm.struct<(i64, ptr)>
// CHECK-COUNT-2:   llvm.insertvalue
//         CHECK:   builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(i64, ptr)> to memref<*xf32>
func.func @construct_unranked_memref_descriptor(%arg0: i64, %arg1: !llvm.ptr) attributes {is_legal} {
  %0 = "test.direct_replacement"(%arg0, %arg1) : (i64, !llvm.ptr) -> (memref<*xf32>)
  "test.legal_op"(%0) : (memref<*xf32>) -> ()
  return
}

// -----

// The argument materializer for unranked MemRef types is called with incorrect
// input types. Make sure that the materializer is skipped and we do not
// generate invalid IR.

// CHECK-LABEL: func @invalid_unranked_memref_descriptor(
//       CHECK:   %[[cast:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i1 to memref<*xf32>
//       CHECK:   "test.legal_op"(%[[cast]])
func.func @invalid_unranked_memref_descriptor(%arg0: i1) attributes {is_legal} {
  %0 = "test.direct_replacement"(%arg0) : (i1) -> (memref<*xf32>)
  "test.legal_op"(%0) : (memref<*xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: llvm.func @simple_func_conversion(
//  CHECK-SAME:     %[[arg0:.*]]: i64) -> i64
//       CHECK:   llvm.return %[[arg0]] : i64
func.func @simple_func_conversion(%arg0: i64) -> i64 {
  return %arg0 : i64
}

// -----

// CHECK-LABEL: llvm.func @one_to_n_argument_conversion(
//  CHECK-SAME:     %[[arg0:.*]]: i18, %[[arg1:.*]]: i18)
//       CHECK:   %[[cast:.*]] = builtin.unrealized_conversion_cast %[[arg0]], %[[arg1]] : i18, i18 to i17
//       CHECK:   "test.legal_op"(%[[cast]]) : (i17) -> ()
func.func @one_to_n_argument_conversion(%arg0: i17) {
  "test.legal_op"(%arg0) : (i17) -> ()
  return
}

// CHECK: llvm.func @caller(%[[arg0:.*]]: i18, %[[arg1:.*]]: i18)
// CHECK:   llvm.call @one_to_n_argument_conversion(%[[arg0]], %[[arg1]]) : (i18, i18) -> ()
func.func @caller(%arg0: i17) {
  func.call @one_to_n_argument_conversion(%arg0) : (i17) -> ()
  return
}

// -----

// CHECK-LABEL: llvm.func @one_to_n_return_conversion(
//  CHECK-SAME:     %[[arg0:.*]]: i18, %[[arg1:.*]]: i18) -> !llvm.struct<(i18, i18)>
//       CHECK:   %[[p1:.*]] = llvm.mlir.poison : !llvm.struct<(i18, i18)>
//       CHECK:   %[[p2:.*]] = llvm.insertvalue %[[arg0]], %[[p1]][0] : !llvm.struct<(i18, i18)>
//       CHECK:   %[[p3:.*]] = llvm.insertvalue %[[arg1]], %[[p2]][1] : !llvm.struct<(i18, i18)>
//       CHECK:   llvm.return %[[p3]]
func.func @one_to_n_return_conversion(%arg0: i17) -> i17 {
  return %arg0 : i17
}

// CHECK: llvm.func @caller(%[[arg0:.*]]: i18, %[[arg1:.*]]: i18)
// CHECK:   %[[res:.*]] = llvm.call @one_to_n_return_conversion(%[[arg0]], %[[arg1]]) : (i18, i18) -> !llvm.struct<(i18, i18)>
// CHECK:   %[[e0:.*]] = llvm.extractvalue %[[res]][0] : !llvm.struct<(i18, i18)>
// CHECK:   %[[e1:.*]] = llvm.extractvalue %[[res]][1] : !llvm.struct<(i18, i18)>
// CHECK:   %[[i0:.*]] = llvm.mlir.poison : !llvm.struct<(i18, i18)>
// CHECK:   %[[i1:.*]] = llvm.insertvalue %[[e0]], %[[i0]][0] : !llvm.struct<(i18, i18)>
// CHECK:   %[[i2:.*]] = llvm.insertvalue %[[e1]], %[[i1]][1] : !llvm.struct<(i18, i18)>
// CHECK:   llvm.return %[[i2]]
func.func @caller(%arg0: i17) -> (i17) {
  %res = func.call @one_to_n_return_conversion(%arg0) : (i17) -> (i17)
  return %res : i17
}

// -----

// CHECK-LABEL: llvm.func @multi_return(
//  CHECK-SAME:     %[[arg0:.*]]: i18, %[[arg1:.*]]: i18, %[[arg2:.*]]: i1) -> !llvm.struct<(i18, i18, i1)>
//       CHECK:   %[[p1:.*]] = llvm.mlir.poison : !llvm.struct<(i18, i18, i1)>
//       CHECK:   %[[p2:.*]] = llvm.insertvalue %[[arg0]], %[[p1]][0] : !llvm.struct<(i18, i18, i1)>
//       CHECK:   %[[p3:.*]] = llvm.insertvalue %[[arg1]], %[[p2]][1] : !llvm.struct<(i18, i18, i1)>
//       CHECK:   %[[p4:.*]] = llvm.insertvalue %[[arg2]], %[[p3]][2] : !llvm.struct<(i18, i18, i1)>
//       CHECK:   llvm.return %[[p4]]
func.func @multi_return(%arg0: i17, %arg1: i1) -> (i17, i1) {
  return %arg0, %arg1 : i17, i1
}

// CHECK: llvm.func @caller(%[[arg0:.*]]: i1, %[[arg1:.*]]: i18, %[[arg2:.*]]: i18)
// CHECK:   %[[res:.*]] = llvm.call @multi_return(%[[arg1]], %[[arg2]], %[[arg0]]) : (i18, i18, i1) -> !llvm.struct<(i18, i18, i1)>
// CHECK:   %[[e0:.*]] = llvm.extractvalue %[[res]][0] : !llvm.struct<(i18, i18, i1)>
// CHECK:   %[[e1:.*]] = llvm.extractvalue %[[res]][1] : !llvm.struct<(i18, i18, i1)>
// CHECK:   %[[e2:.*]] = llvm.extractvalue %[[res]][2] : !llvm.struct<(i18, i18, i1)>
// CHECK:   %[[i0:.*]] = llvm.mlir.poison : !llvm.struct<(i18, i18, i1, i18, i18)>
// CHECK:   %[[i1:.*]] = llvm.insertvalue %[[e0]], %[[i0]][0]
// CHECK:   %[[i2:.*]] = llvm.insertvalue %[[e1]], %[[i1]][1]
// CHECK:   %[[i3:.*]] = llvm.insertvalue %[[e2]], %[[i2]][2]
// CHECK:   %[[i4:.*]] = llvm.insertvalue %[[e0]], %[[i3]][3]
// CHECK:   %[[i5:.*]] = llvm.insertvalue %[[e1]], %[[i4]][4]
// CHECK:   llvm.return %[[i5]]
func.func @caller(%arg0: i1, %arg1: i17) -> (i17, i1, i17) {
  %res:2 = func.call @multi_return(%arg1, %arg0) : (i17, i1) -> (i17, i1)
  return %res#0, %res#1, %res#0 : i17, i1, i17
}

// -----

// CHECK-LABEL: llvm.func @branch(
//  CHECK-SAME:     %[[arg0:.*]]: i1, %[[arg1:.*]]: i18, %[[arg2:.*]]: i18)
//       CHECK:   llvm.br ^[[bb1:.*]](%[[arg1]], %[[arg2]], %[[arg0]] : i18, i18, i1)
//       CHECK: ^[[bb1]](%[[arg3:.*]]: i18, %[[arg4:.*]]: i18, %[[arg5:.*]]: i1):
//       CHECK:   llvm.cond_br %[[arg5]], ^[[bb1]](%[[arg1]], %[[arg2]], %[[arg5]] : i18, i18, i1), ^[[bb2:.*]](%[[arg3]], %[[arg4]] : i18, i18)
//       CHECK: ^bb2(%{{.*}}: i18, %{{.*}}: i18):
//       CHECK:   llvm.return
func.func @branch(%arg0: i1, %arg1: i17) {
  cf.br ^bb1(%arg1, %arg0: i17, i1)
^bb1(%arg2: i17, %arg3: i1):
  cf.cond_br %arg3, ^bb1(%arg1, %arg3 : i17, i1), ^bb2(%arg2 : i17)
^bb2(%arg4: i17):
  return
}
