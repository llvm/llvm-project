// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" -drop-equivalent-buffer-results -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs unknown-type-conversion=identity-layout-map bufferize-function-boundaries" -split-input-file -o /dev/null

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A0:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[A1:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[t0:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[t1:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func.func @insert_slice_fun(
    %A0 : tensor<?xf32> {bufferization.writable = false},
    %A1 : tensor<?xf32> {bufferization.writable = true},
    %t0 : tensor<4xf32> {bufferization.writable = false},
    %t1 : tensor<4xf32> {bufferization.writable = true})
  ->  (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
{
  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC3:.*]] = memref.alloc
  //      CHECK: memref.copy %[[A0]], %[[REALLOC3]]
  //      CHECK: %[[SV_A0:.*]] = memref.subview %[[REALLOC3]]
  //      CHECK: memref.copy %[[t0]], %[[SV_A0]]
  %r0 = tensor.insert_slice %t0 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC2:.*]] = memref.alloc
  //      CHECK: memref.copy %[[A0]]
  //      CHECK: %[[SV_A0_2:.*]] = memref.subview %[[REALLOC2]]
  //      CHECK: memref.copy %[[t1]], %[[SV_A0_2]]
  %r1 = tensor.insert_slice %t1 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Still alloc the large tensor because %A1 is read after. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC1:.*]] = memref.alloc
  //      CHECK: memref.copy %[[A1]]
  //      CHECK: %[[SV_A1:.*]] = memref.subview %[[REALLOC1]]
  //      CHECK: memref.copy %[[t0]], %[[SV_A1]]
  %r2 = tensor.insert_slice %t0 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Do not realloc the large tensor. Copy the tensor.extract_slice.
  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A1_2:.*]] = memref.subview %[[A1]]
  //      CHECK: memref.copy %[[t1]], %[[SV_A1_2]]
  %r3 = tensor.insert_slice %t1 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return %[[REALLOC3]], %[[REALLOC2]], %[[REALLOC1]] :
  // CHECK-SAME:   memref<?xf32>, memref<?xf32>, memref<?xf32>
  return %r0, %r1, %r2, %r3: tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func.func @insert_slice_fun(
    %A : tensor<?xf32> {bufferization.writable = true},
    %t : tensor<4xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  %f0 = arith.constant 0.0 : f32

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A:.*]] = memref.subview %[[A]]
  //      CHECK: memref.copy %[[t]], %[[SV_A]]
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  /// Overwrite A inplace.
  //      CHECK: linalg.fill ins({{.*}}{{.*}}outs(%[[A]]
  %r1 = linalg.fill ins(%f0 : f32) outs(%r0 : tensor<?xf32>) -> tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r1: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func.func @insert_slice_fun(
    %A : tensor<?xf32> {bufferization.writable = true},
    %t : tensor<4xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  %f0 = arith.constant 0.0 : f32

  //      CHECK: linalg.fill ins({{.*}}{{.*}}outs(%[[A]]
  %r0 = linalg.fill ins(%f0 : f32) outs(%A : tensor<?xf32>) -> tensor<?xf32>

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A:.*]] = memref.subview %[[A]]
  /// Overwrite A inplace by copying into the subview.
  //      CHECK: memref.copy %[[t]], %[[SV_A]]
  %r1 = tensor.insert_slice %t into %r0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r1: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun_not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func.func @insert_slice_fun_not_inplace(
    %A : tensor<?xf32> {bufferization.writable = false},
    %t : tensor<4xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) {alignment = 128 : i64} : memref<?xf32>
  //      CHECK: memref.copy %[[A]], %[[ALLOC]] : memref<?xf32{{.*}} to memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32>
  //      CHECK: memref.copy %[[t]], %[[SV]] : memref<4xf32, #map> to memref<4xf32>
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //     CHECK: return %{{.*}} : memref<?xf32>
  return %r0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @tensor_cast_not_in_place(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32{{.*}}>, %[[B:.*]]: memref<?xf32{{.*}}>
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   memref.copy %[[A]], %[[alloc]]
//       CHECK:   %[[subview:.*]] = memref.subview %[[A]][{{.*}}] [4] [1] : {{.*}} to memref<4xf32
//       CHECK:   memref.copy %[[alloc]], %[[subview]]
func.func @tensor_cast_not_in_place(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = false}, %idx: index)
  -> (tensor<?xf32>)
{
  %r0 = tensor.cast %A : tensor<?xf32> to tensor<4xf32>
  %r1 = tensor.insert_slice %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_op
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, {{.*}}>, %[[s:.*]]: f32, %[[i:.*]]: index
func.func @insert_op(%t1 : tensor<?xf32> {bufferization.writable = true},
                     %s : f32, %i : index) -> tensor<?xf32> {
  // CHECK: memref.store %[[s]], %[[t1]][%[[i]]]
  %0 = tensor.insert %s into %t1[%i] : tensor<?xf32>
  // CHECK: return
  return %0 : tensor<?xf32>
}

// -----

// A regression test to make sure that we handle rank-reducing extract_slice
// correctly.

// CHECK-LABEL: func @rank_reducing
func.func @rank_reducing(
    %i: index, %j: index,
    %arg0: tensor<8x18x32xf32>)
      -> tensor<?x1x6x8xf32> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = bufferization.alloc_tensor() : tensor<4x1x6x8xf32>
  %1 = tensor.cast %0 : tensor<4x1x6x8xf32> to tensor<?x1x6x8xf32>
  %2 = bufferization.alloc_tensor() : tensor<1x6x8xf32>
  %5 = scf.for %arg7 = %c0 to %c32 step %c8 iter_args(%arg8 = %1) -> (tensor<?x1x6x8xf32>) {
    %7 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg7)
    %8 = tensor.extract_slice %arg0[%i, %j, %arg7] [1, 6, 8] [1, 1, 1] : tensor<8x18x32xf32> to tensor<1x6x8xf32>
    %9 = scf.for %arg9 = %c0 to %c6 step %c1 iter_args(%arg10 = %2) -> (tensor<1x6x8xf32>) {
      %11 = tensor.extract_slice %8[0, %arg9, 0] [1, 1, 8] [1, 1, 1] : tensor<1x6x8xf32> to tensor<1x1x8xf32>
      %12 = tensor.insert_slice %11 into %arg10[0, %arg9, 0] [1, 1, 8] [1, 1, 1] : tensor<1x1x8xf32> into tensor<1x6x8xf32>
      scf.yield %12 : tensor<1x6x8xf32>
    }
    %10 = tensor.insert_slice %9 into %arg8[%7, 0, 0, 0] [1, 1, 6, 8] [1, 1, 1, 1] : tensor<1x6x8xf32> into tensor<?x1x6x8xf32>
    scf.yield %10 : tensor<?x1x6x8xf32>
  }
  return %5: tensor<?x1x6x8xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

// CHECK-LABEL: func.func @rank_reducing_parallel_insert_slice
func.func @rank_reducing_parallel_insert_slice(%in: tensor<100xf32>, %out: tensor<200x100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  // CHECK: scf.foreach_thread {{.*}} {
  %result = scf.foreach_thread (%thread_idx) in (%num_threads) shared_outs (%o = %out) -> tensor<200x100xf32> {
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      scf.foreach_thread.perform_concurrently {
        // CHECK: memref.subview %{{.*}}[%{{.*}}] [1] [1] : memref<100xf32, #[[$MAP0]]> to memref<1xf32, #[[$MAP0]]>
        // CHECK: memref.subview %{{.*}}[1, %{{.*}}] [1, 1] [1, 1] : memref<200x100xf32, #[[$MAP1]]> to memref<1xf32, #[[$MAP0]]>
        tensor.parallel_insert_slice %1 into %o[1, %thread_idx][1, 1][1, 1] :
          tensor<1xf32> into tensor<200x100xf32>
      }
  }
  // CHECK: }
  return
}

// -----

// CHECK-LABEL: func @dealloc_generate_buffer
func.func @dealloc_generate_buffer(%arg: tensor<*xf32>, %sz: index, %idx: index)
  -> index
{
  // CHECK: memref.alloc
  // CHECK: scf.parallel
  // CHECK: memref.load
  // CHECK: memref.dealloc
  %0 = tensor.generate %sz {
  ^bb0(%i : index):
    %elem = tensor.dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  %r = tensor.extract %0[%idx] : tensor<?xindex>
  return %r : index
}

// -----

// CHECK-LABEL: func @dealloc_pad_buffer
func.func @dealloc_pad_buffer(%t1: tensor<?x10xindex>, %l2: index, %h1: index,
                              %h2: index, %idx: index) -> index {
  // CHECK: memref.alloc
  // CHECK: scf.parallel
  // CHECK: memref.load
  // CHECK: memref.dealloc
  %0 = tensor.pad %t1 low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %m = arith.muli %arg0, %arg1 : index
    tensor.yield %m : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  %r = tensor.extract %0[%idx, %idx] : tensor<?x?xindex>
  return %r : index
}
