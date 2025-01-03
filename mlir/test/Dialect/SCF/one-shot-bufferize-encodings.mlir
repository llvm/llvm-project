// RUN: mlir-opt %s -one-shot-bufferize="use-encoding-for-memory-space allow-return-allocs-from-loops allow-unknown-ops" -allow-unregistered-dialect -split-input-file | FileCheck %s

// Here and below, unknown op 'some.use' will force 'bufferization.to_tensor' operations to remain in the body,
// allowing us to check that the encoding on the '%iter' tensor is correctly preserved.

func.func @scf_for_iter_arg(%arg0: tensor<128xf32, 1>, %arg1: index, %arg2: index, %arg3: index) -> tensor<128xf32, 1> {
  %0 = scf.for %i = %arg1 to %arg2 step %arg3 iter_args(%iter = %arg0) -> tensor<128xf32, 1> {
    %0 = "some.use"(%iter) : (tensor<128xf32, 1>) -> tensor<128xf32, 1>
    scf.yield %0 : tensor<128xf32, 1>
  }
  return %0 : tensor<128xf32, 1>
}

// CHECK-LABEL: func.func @scf_for_iter_arg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32, 1 : i64>, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index)
//       CHECK:     %[[v0:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
//       CHECK:     memref.copy %[[v0]], %[[alloc]] : memref<128xf32, strided<[?], offset: ?>, 1> to memref<128xf32, 1>
//       CHECK:     %[[cast:.+]] = memref.cast %[[alloc]] : memref<128xf32, 1> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v1:.+]] = scf.for %{{.+}} = %[[arg1]] to %[[arg2]] step %[[arg3]] iter_args(%[[arg6:.+]] = %[[cast]]) -> (memref<128xf32, strided<[?], offset: ?>, 1>)
//  CHECK-NEXT:       %[[v3:.+]] = bufferization.to_tensor %[[arg6]] : memref<128xf32, strided<[?], offset: ?>, 1> to tensor<128xf32, 1 : i64>
//  CHECK-NEXT:       %[[v4:.+]] = "some.use"(%[[v3]]) : (tensor<128xf32, 1 : i64>) -> tensor<128xf32, 1 : i64>
//  CHECK-NEXT:       %[[v5:.+]] = bufferization.to_memref %[[v4]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//  CHECK-NEXT:       scf.yield %[[v5]] : memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v2:.+]] = bufferization.to_tensor %[[v1]] : memref<128xf32, strided<[?], offset: ?>, 1> to tensor<128xf32, 1 : i64>
//       CHECK:     return %[[v2]] : tensor<128xf32, 1 : i64>

// -----

func.func @scf_forall(
    %idx: index,
    %idx2: index,
    %arg1: tensor<?xf32, 1>,
    %arg2: tensor<?xf32, 1>) -> (tensor<?xf32, 1>) {
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = scf.forall (%arg3) in (%idx2) shared_outs(%o = %arg2) -> (tensor<?xf32, 1>) {
      %8 = "some.use"(%o) : (tensor<?xf32, 1>) -> tensor<?xf32, 1>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %o[5] [%idx] [%c1] :
          tensor<?xf32, 1> into tensor<?xf32, 1>
      }
  }
  return %2 : tensor<?xf32, 1>
}

// CHECK-LABEL: func.func @scf_forall
//       CHECK:     scf.forall
//       CHECK:       %[[v2:.+]] = bufferization.to_tensor %{{.+}} : memref<?xf32, 1> to tensor<?xf32, 1 : i64>
//       CHECK:       %[[v3:.+]] = "some.use"(%[[v2]]) : (tensor<?xf32, 1 : i64>) -> tensor<?xf32, 1 : i64>
//       CHECK:       bufferization.to_memref %[[v3]] : tensor<?xf32, 1 : i64> to memref<?xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v1:.+]] = bufferization.to_tensor %{{.+}} : memref<?xf32, 1> to tensor<?xf32, 1 : i64>
//       CHECK:     return %[[v1]] : tensor<?xf32, 1 : i64>

// -----

func.func @scf_execute_region(%arg0: tensor<128xf32, 1>) -> tensor<128xf32, 1> {
  %0 = scf.execute_region -> tensor<128xf32, 1> {
    scf.yield %arg0 : tensor<128xf32, 1>
  }
  %1 = "some.use"(%0) : (tensor<128xf32, 1>) -> tensor<128xf32, 1>
  return %1 : tensor<128xf32, 1>
}

// CHECK-LABEL: func.func @scf_execute_region
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32, 1 : i64>)
//       CHECK:     %[[v0:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v1:.+]] = scf.execute_region -> memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:       scf.yield %[[v0]] : memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v2:.+]] = bufferization.to_tensor %[[v1]] : memref<128xf32, strided<[?], offset: ?>, 1> to tensor<128xf32, 1 : i64>
//       CHECK:     %[[v3:.+]] = "some.use"(%[[v2]]) : (tensor<128xf32, 1 : i64>) -> tensor<128xf32, 1 : i64>
//       CHECK:     return %[[v3]] : tensor<128xf32, 1 : i64>
