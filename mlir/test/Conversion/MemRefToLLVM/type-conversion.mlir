// RUN: mlir-opt %s -test-llvm-legalize-patterns -split-input-file

// Test the argument materializer for ranked MemRef types.

//   CHECK-LABEL: func @construct_ranked_memref_descriptor(
//         CHECK:   llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-COUNT-7:   llvm.insertvalue
//         CHECK:   builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<5x4xf32>
func.func @construct_ranked_memref_descriptor(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
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
func.func @invalid_ranked_memref_descriptor(%arg0: i1) {
  %0 = "test.direct_replacement"(%arg0) : (i1) -> (memref<5x4xf32>)
  "test.legal_op"(%0) : (memref<5x4xf32>) -> ()
  return
}

// -----

// Test the argument materializer for unranked MemRef types.

//   CHECK-LABEL: func @construct_unranked_memref_descriptor(
//         CHECK:   llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK-COUNT-2:   llvm.insertvalue
//         CHECK:   builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(i64, ptr)> to memref<*xf32>
func.func @construct_unranked_memref_descriptor(%arg0: i64, %arg1: !llvm.ptr) {
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
func.func @invalid_unranked_memref_descriptor(%arg0: i1) {
  %0 = "test.direct_replacement"(%arg0) : (i1) -> (memref<*xf32>)
  "test.legal_op"(%0) : (memref<*xf32>) -> ()
  return
}
