// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize=allow-return-allocs -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-allocs fully-dynamic-layout-maps=0" -split-input-file | FileCheck %s --check-prefix=CHECK-NO-LAYOUT-MAP

// CHECK-LABEL: func @transfer_read(%{{.*}}: memref<?xf32, #map>) -> vector<4xf32> {
// CHECK-NO-LAYOUT-MAP-LABEL: func @transfer_read(%{{.*}}: memref<?xf32>) -> vector<4xf32>
func @transfer_read(
    %A : tensor<?xf32> {linalg.inplaceable = false})
  -> (vector<4xf32>)
{
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

//       CHECK: %[[RES:.*]] = vector.transfer_read {{.*}} : memref<?xf32, #{{.*}}>, vector<4xf32>
  %0 = vector.transfer_read %A[%c0], %f0 : tensor<?xf32>, vector<4xf32>

//       CHECK: return %[[RES]] : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @fill_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
// CHECK-NO-LAYOUT-MAP-LABEL: func @fill_inplace(%{{.*}}: memref<?xf32>) {
func @fill_inplace(
    %A : tensor<?xf32> {linalg.inplaceable = true})
  -> tensor<?xf32>
{
  //     CHECK: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0 : f32

  /// Inplaceable, no alloc
  // CHECK-NOT: alloc
  //     CHECK: linalg.fill ins(%[[F0]] : f32) outs(%[[A]] : memref<?xf32, #[[$map_1d_dyn]]>)
  %r = linalg.fill ins(%f0 : f32) outs(%A : tensor<?xf32>) -> tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @tensor_extract(%{{.*}}: memref<?xf32, #{{.*}}>) -> f32 {
func @tensor_extract(%A : tensor<?xf32> {linalg.inplaceable = false}) -> (f32) {
  %c0 = arith.constant 0 : index

//       CHECK: %[[RES:.*]] = memref.load {{.*}} : memref<?xf32, #{{.*}}>
  %0 = tensor.extract %A[%c0] : tensor<?xf32>

//       CHECK: return %[[RES]] : f32
  return %0 : f32
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

/// No linalg.inplaceable flag, must allocate.
// CHECK-LABEL: func @not_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>) -> memref<?xf32> {
// CHECK-NO-LAYOUT-MAP-LABEL: func @not_inplace(%{{.*}}: memref<?xf32>) -> memref<?xf32>
func @not_inplace(
    %A : tensor<?xf32> {linalg.inplaceable = false})
  -> tensor<?xf32>
{
  //     CHECK: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0 : f32

  //     CHECK: %[[D0:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, #[[$map_1d_dyn]]>
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[D0]]) {alignment = 128 : i64} : memref<?xf32>
  //     CHECK: linalg.fill ins(%[[F0]] : f32) outs(%[[ALLOC]] : memref<?xf32>)
  %r = linalg.fill ins(%f0 : f32) outs(%A : tensor<?xf32>) -> tensor<?xf32>

  // CHECK-NOT: dealloc
  //     CHECK: return %[[ALLOC]] : memref<?xf32>
  return %r: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_2d_dyn:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

// CHECK-LABEL: func @not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?x?xf32, #[[$map_2d_dyn]]>) {
// CHECK-NO-LAYOUT-MAP-LABEL: func @not_inplace(%{{.*}}: memref<?x?xf32>) {
func @not_inplace(
    %A : tensor<?x?xf32> {linalg.inplaceable = true})
  -> tensor<?x?xf32>
{
  %f0 = arith.constant 0.0 : f32

  /// Cross-op multiple uses of %A, the first op which has interfering reads must alloc.
  //       CHECK: %[[ALLOC:.*]] = memref.alloc
  //       CHECK: linalg.fill ins({{.*}}{{.*}}outs(%[[ALLOC]]
  %f = linalg.fill ins(%f0 : f32) outs(%A : tensor<?x?xf32>) -> tensor<?x?xf32>

  /// The second op has no interfering reads and can reuse.
  //   CHECK-NOT: alloc
  //       CHECK: linalg.matmul ins(%[[ALLOC]], %[[ALLOC]]{{.*}}) outs(%[[A]]
  %r = linalg.matmul  ins(%f, %f: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  //     CHECK: memref.dealloc %[[ALLOC]]
  //     CHECK: return
  // CHECK-NOT: tensor
  return %r: tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @not_inplace
func @not_inplace(%A : tensor<?x?xf32> {linalg.inplaceable = true}) -> tensor<?x?xf32> {
  /// Within op multiple uses of %A, must alloc.
  // CHECK: alloc
  %r = linalg.matmul  ins(%A, %A: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  // CHECK-NOT: dealloc
  return %r: tensor<?x?xf32>
}
// -----

// CHECK-LABEL: func @vec_inplace
func @vec_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %vec : vector<4xf32>)
    -> tensor<?xf32>
{
  %c0 = arith.constant 0 : index

  // CHECK-NOT: alloc
  %r = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @vec_not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
func @vec_not_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %vec : vector<4xf32>)
    -> (tensor<?xf32>, tensor<?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  /// Cross-op multiple uses of %A, the first vector.transfer which has interfering reads must alloc.
  //      CHECK: %[[ALLOC:.*]] = memref.alloc
  //      CHECK: memref.copy {{.*}}, %[[ALLOC]]
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[ALLOC]]
  %r0 = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>

  /// The second vector.transfer has no interfering reads and can reuse the buffer.
  //  CHECK-NOT: alloc
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[A]]
  %r1 = vector.transfer_write %vec, %A[%c1] : vector<4xf32>, tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A0:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[A1:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[t0:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[t1:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func @insert_slice_fun(%A0 : tensor<?xf32> {linalg.inplaceable = false},
                       %A1 : tensor<?xf32> {linalg.inplaceable = true},
                       %t0 : tensor<4xf32> {linalg.inplaceable = false},
                       %t1 : tensor<4xf32> {linalg.inplaceable = true})
  ->  (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
{
  // Hoisted allocs.
  //      CHECK: %[[REALLOC1:.*]] = memref.alloc
  //      CHECK: %[[REALLOC2:.*]] = memref.alloc
  //      CHECK: %[[REALLOC3:.*]] = memref.alloc

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: memref.copy %[[A0]], %[[REALLOC3]]
  //      CHECK: %[[SV_A0:.*]] = memref.subview %[[REALLOC3]]
  //      CHECK: memref.copy %[[t0]], %[[SV_A0]]
  %r0 = tensor.insert_slice %t0 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: memref.copy %[[A0]]
  //      CHECK: %[[SV_A0_2:.*]] = memref.subview %[[REALLOC2]]
  //      CHECK: memref.copy %[[t1]], %[[SV_A0_2]]
  %r1 = tensor.insert_slice %t1 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Still alloc the large tensor because %A1 is read after. Copy the tensor.extract_slice.
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
func @insert_slice_fun(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %t : tensor<4xf32> {linalg.inplaceable = false})
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
func @insert_slice_fun(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %t : tensor<4xf32> {linalg.inplaceable = false})
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
func @insert_slice_fun_not_inplace(
    %A : tensor<?xf32> {linalg.inplaceable = false},
    %t : tensor<4xf32> {linalg.inplaceable = false})
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

//===----------------------------------------------------------------------===//
// Simple loop cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @scf_for_yield_only
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
func @scf_for_yield_only(%A : tensor<?xf32> {linalg.inplaceable = false},
                         %B : tensor<?xf32> {linalg.inplaceable = true},
                         %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   %[[CASTED:.*]] = memref.cast %[[ALLOC_FOR_A]]
  //     CHECK:   memref.copy %[[A]], %[[ALLOC_FOR_A]]

  // The first scf.for remains but just turns into dead code.
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  // The second scf.for remains but just turns into dead code.
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //     CHECK:   return %[[CASTED]] : memref<?xf32, #[[$map_1d_dyn]]>
  // CHECK-NOT:   dealloc
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// Ensure that the function bufferizes without error. This tests pre-order
// traversal of scf.for loops during bufferization. No need to check the IR,
// just want to make sure that it does not crash.

// CHECK-LABEL: func @nested_scf_for
func @nested_scf_for(%A : tensor<?xf32> {linalg.inplaceable = true},
                     %v : vector<5xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %r1 = scf.for %i = %c0 to %c10 step %c1 iter_args(%B = %A) -> tensor<?xf32> {
    %r2 = scf.for %j = %c0 to %c10 step %c1 iter_args(%C = %B) -> tensor<?xf32> {
      %w = vector.transfer_write %v, %C[%c0] : vector<5xf32>, tensor<?xf32>
      scf.yield %w : tensor<?xf32>
    }
    scf.yield %r2 : tensor<?xf32>
  }
  return %r1 : tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func @scf_for_with_tensor.insert_slice(
   %A : tensor<?xf32> {linalg.inplaceable = false},
   %B : tensor<?xf32> {linalg.inplaceable = true},
   %C : tensor<4xf32> {linalg.inplaceable = false},
   %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   %[[CASTED:.*]] = memref.cast %[[ALLOC_FOR_A]]
  //     CHECK:   memref.copy %[[A]], %[[ALLOC_FOR_A]]

  //     CHECK: %[[svA:.*]] = memref.subview %[[ALLOC_FOR_A]][0] [4] [1]
  //     CHECK: %[[svB:.*]] = memref.subview %[[B]][0] [4] [1]

  //     CHECK:   scf.for {{.*}}
  // CHECK-NOT: iter_args
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // %ttA bufferizes to direct copy of %BUFFER_CAST_C into %svA
    //     CHECK: memref.copy %[[C]], %[[svA]]
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // %ttB bufferizes to direct copy of %BUFFER_CAST_C into %BUFFER_CAST_B
    //     CHECK:   memref.copy %[[C]], %[[svB]]
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  //     CHECK:  return %[[CASTED]] : memref<?xf32, #[[$map_1d_dyn]]>
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Cross function boundary cases.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @execute_region_with_conflict(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @execute_region_with_conflict(%t1 : tensor<?xf32> {linalg.inplaceable = "true"})
    -> (f32, tensor<?xf32>, f32)
{
  %f1 = arith.constant 0.0 : f32
  %idx = arith.constant 7 : index

  // scf.execute_region is canonicalized away after bufferization. So just the
  // memref.store is left over.

  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK: memref.copy %[[m1]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]][%{{.*}}]
  %0, %1, %2 = scf.execute_region -> (f32, tensor<?xf32>, f32) {
    %t2 = tensor.insert %f1 into %t1[%idx] : tensor<?xf32>
    scf.yield %f1, %t2, %f1 : f32, tensor<?xf32>, f32
  }

  // CHECK: %[[load:.*]] = memref.load %[[m1]]
  %3 = tensor.extract %t1[%idx] : tensor<?xf32>

  // CHECK: return %{{.*}}, %[[casted]], %[[load]] : f32, memref<?xf32, #{{.*}}>, f32
  return %0, %1, %3 : f32, tensor<?xf32>, f32
}

// -----

//      CHECK: func @matmul(
// CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: memref<128x256xf32>
// CHECK-SAME:   %[[B:[0-9a-zA-Z]*]]: memref<256x192xf32>
// CHECK-SAME:   %[[C:[0-9a-zA-Z]*]]: memref<128x192xf32>
func @matmul(
    %A: tensor<128x256xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %B: tensor<256x192xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %C: tensor<128x192xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
      -> tensor<128x192xf32> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c192 = arith.constant 192 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // Hoisted alloc.
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<8x16xf32>

  // CHECK: scf.for %[[I:.*]] =
  %0 = scf.for %arg3 = %c0 to %c128 step %c8 iter_args(%arg4 = %C) -> (tensor<128x192xf32>) {
    %1 = tensor.extract_slice %A[%arg3, 0] [8, 256] [1, 1] :
      tensor<128x256xf32> to tensor<8x256xf32>

    // CHECK: scf.for %[[J:.*]] =
    %2 = scf.for %arg5 = %c0 to %c192 step %c16 iter_args(%arg6 = %arg4) -> (tensor<128x192xf32>) {
      %3 = tensor.extract_slice %B[0, %arg5] [256, 16] [1, 1] :
        tensor<256x192xf32> to tensor<256x16xf32>

      // %4 does not match an insert_slice, it cannot be bufferized inplace and needs to alloc.
      %4 = tensor.extract_slice %C[%arg3, %arg5] [8, 16] [1, 1] :
        tensor<128x192xf32> to tensor<8x16xf32>

      // linalg.fill is inplace.
      // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC]] : memref<8x16xf32>)
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<8x16xf32>) -> tensor<8x16xf32>

      // CHECK: scf.for %[[K:.*]] =
      %6 = scf.for %arg7 = %c0 to %c256 step %c32 iter_args(%arg8 = %5) -> (tensor<8x16xf32>) {
        %8 = tensor.extract_slice %1[0, %arg7] [8, 32] [1, 1] :
          tensor<8x256xf32> to tensor<8x32xf32>
        %9 = tensor.extract_slice %3[%arg7, 0] [32, 16] [1, 1] :
          tensor<256x16xf32> to tensor<32x16xf32>

        // linalg.matmul is inplace as well as the enclosing scf.for.
        // CHECK: linalg.matmul ins({{.*}} outs(%[[ALLOC]]
        %10 = linalg.matmul ins(%8, %9 : tensor<8x32xf32>, tensor<32x16xf32>)
                           outs(%arg8 : tensor<8x16xf32>)
          -> tensor<8x16xf32>
        scf.yield %10 : tensor<8x16xf32>
      }

      // insert_slice is inplace but its source comes from an equivalent buffer
      // that is not in place. So we must insert a copy of the small buffer into
      // the bigger buffer.
      // CHECK: %[[T:.*]] = memref.subview %[[C]][%[[I]], %[[J]]] [8, 16] [1, 1]
      // CHECK: memref.copy %[[ALLOC]], %[[T]]
      %7 = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [8, 16] [1, 1] :
        tensor<8x16xf32> into tensor<128x192xf32>

      // CHECK: memref.dealloc %[[ALLOC]]
      scf.yield %7 : tensor<128x192xf32>
    }
    scf.yield %2 : tensor<128x192xf32>
  }
  return %0 : tensor<128x192xf32>
}

// -----

// CHECK-LABEL: func @tensor_cast_not_in_place(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32{{.*}}>, %[[B:.*]]: memref<?xf32{{.*}}>
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   memref.copy %[[A]], %[[alloc]]
//       CHECK:   %[[subview:.*]] = memref.subview %[[A]][{{.*}}] [4] [1] : {{.*}} to memref<4xf32
//       CHECK:   memref.copy %[[alloc]], %[[subview]]
func @tensor_cast_not_in_place(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32> {linalg.inplaceable = false}, %idx: index)
  -> (tensor<?xf32>)
{
  %r0 = tensor.cast %A : tensor<?xf32> to tensor<4xf32>
  %r1 = tensor.insert_slice %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r1 : tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Insertion point cases.
//===----------------------------------------------------------------------===//

/// These tests just check the produced IR is valid and does not have dominance
/// errors in the def-use chains.

// CHECK-LABEL: func @dominance_violation_bug_1
func @dominance_violation_bug_1(
    %A : tensor<?x?xf32> {linalg.inplaceable = false},
    %idx : index)
  -> tensor<?x?xf32>
{
  %f0 = arith.constant 0.0 : f32

  %sA = tensor.extract_slice %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssA = tensor.extract_slice %sA[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FA = linalg.fill ins(%f0 : f32) outs(%ssA : tensor<4x4xf32>) -> tensor<4x4xf32>
  %rsA = tensor.insert_slice %FA into %sA[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rA = tensor.insert_slice %rsA into %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  return %rA : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inplace(
//  CHECK-SAME:     %[[cond:.*]]: i1, %[[t1:.*]]: memref<?xf32{{.*}}>, %[[v:.*]]: vector
func @scf_if_inplace(%cond: i1,
                     %t1: tensor<?xf32> {linalg.inplaceable = true},
                     %v: vector<5xf32>, %idx: index) -> tensor<?xf32> {

  //      CHECK: scf.if %[[cond]] {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   vector.transfer_write %[[v]], %[[t1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %r = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %t1 : tensor<?xf32>
  } else {
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inside_scf_for
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   scf.for %{{.*}} = %[[c0]] to %[[c10]] step %[[c1]] {
//       CHECK:     scf.if %{{.*}} {
//       CHECK:     } else {
//       CHECK:       vector.transfer_write
//       CHECK:     }
//       CHECK:   }
func @scf_if_inside_scf_for(%t1: tensor<?xf32> {linalg.inplaceable = true},
                      %v: vector<5xf32>, %idx: index,
                      %cond: i1) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%bb = %t1) -> (tensor<?xf32>) {
    %r2 = scf.if %cond -> (tensor<?xf32>) {
      scf.yield %bb : tensor<?xf32>
    } else {
      %t2 = vector.transfer_write %v, %bb[%idx] : vector<5xf32>, tensor<?xf32>
      scf.yield %t2 : tensor<?xf32>
    }
    scf.yield %r2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_non_equiv_yields(
//  CHECK-SAME:     %[[cond:.*]]: i1, %[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>) -> memref<{{.*}}>
func @scf_if_non_equiv_yields(
    %b : i1,
    %A : tensor<4xf32> {linalg.inplaceable = false},
    %B : tensor<4xf32> {linalg.inplaceable = false})
  -> tensor<4xf32>
{
  // CHECK: %[[r:.*]] = arith.select %[[cond]], %[[A]], %[[B]]
  %r = scf.if %b -> (tensor<4xf32>) {
    scf.yield %A : tensor<4xf32>
  } else {
    scf.yield %B : tensor<4xf32>
  }
  // CHECK: return %[[r]]
  return %r: tensor<4xf32>
}

// -----

// CHECK-LABEL: func @insert_op
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, {{.*}}>, %[[s:.*]]: f32, %[[i:.*]]: index
func @insert_op(%t1 : tensor<?xf32> {linalg.inplaceable = true},
                %s : f32, %i : index) -> tensor<?xf32> {
  // CHECK: memref.store %[[s]], %[[t1]][%[[i]]]
  %0 = tensor.insert %s into %t1[%i] : tensor<?xf32>
  // CHECK: return
  return %0 : tensor<?xf32>
}

// -----

func @gather_like(
    %arg0 : tensor<?x?xf32> {linalg.inplaceable = false},
    %arg1 : tensor<?xi32> {linalg.inplaceable = false},
    %arg2 : tensor<?x?xf32> {linalg.inplaceable = true}) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<?xi32>) outs(%arg2 : tensor<?x?xf32>) {
      ^bb0(%arg3: i32, %arg4 : f32):
        %iv1 = linalg.index 1 : index
        %1 = arith.index_cast %arg3: i32 to index
        %2 = tensor.extract %arg0[%1, %iv1] : tensor<?x?xf32>
        linalg.yield %2 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @gather_like(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32,
//  CHECK-SAME:     %[[ARG1:.+]]: memref<?xi32
//  CHECK-SAME:     %[[ARG2:.+]]: memref<?x?xf32
//  CHECK-SAME:   ) {
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]] :
//  CHECK-SAME:       outs(%[[ARG2]] :
//       CHECK:     %[[YIELD:.+]] = memref.load %[[ARG0]]
//       CHECK:     linalg.yield %[[YIELD]]

// -----

// CHECK-LABEL: func @linalg_op_bufferizes_inplace_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>, %[[t3:.*]]: memref<?x?xf32, #{{.*}}>
func @linalg_op_bufferizes_inplace_with_input(
    %t1: tensor<?x?xf32> {linalg.inplaceable = true},
    %t2: tensor<?xf32> {linalg.inplaceable = true},
    %t3: tensor<?x?xf32> {linalg.inplaceable = true},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[t3]] : {{.*}})
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%t1, %t2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%t3 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
        %add = arith.addf %arg0, %arg1 : f32
        linalg.yield %add : f32
    } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @op_is_reading_but_following_ops_are_not
//  CHECK-SAME:     %[[t0:.*]]: memref<?xf32
func @op_is_reading_but_following_ops_are_not(
    %t0 : tensor<?xf32> {linalg.inplaceable = false},
    %cst : f32)
  -> tensor<?xf32>
{
  // Make sure that a copy is inserted here.
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  // CHECK: memref.copy %[[t0]], %[[ALLOC]]
  // CHECK: linalg.generic {{.*}} outs(%[[ALLOC]] : memref
  %r0 =linalg.generic #trait outs (%t0 : tensor<?xf32>) {
      ^bb(%0: f32) :
        %a = arith.addf %cst, %0 : f32
        linalg.yield %a : f32
    } -> (tensor<?xf32>)

  // CHECK: linalg.generic {{.*}} outs(%[[ALLOC]] : memref
  %r1 = linalg.generic #trait outs (%r0 : tensor<?xf32>) {
      ^bb(%0: f32) :
        linalg.yield %cst : f32
    } -> (tensor<?xf32>)

  // CHECK: return %[[ALLOC]]
  return %r1 : tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// InitTensorOp elimination would produce SSA violations for the example below.
//===----------------------------------------------------------------------===//

func @depthwise_conv_1d_nwc_wc(%arg0: index, %arg1: index, %arg2: tensor<8x18x32xf32>)
    -> tensor<?x1x6x8xf32> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %0 = linalg.init_tensor [4, 1, 6, 8] : tensor<4x1x6x8xf32>
  %1 = tensor.cast %0 : tensor<4x1x6x8xf32> to tensor<?x1x6x8xf32>
  %2 = linalg.init_tensor [1, 6, 8] : tensor<1x6x8xf32>
  %3 = scf.for %arg3 = %c0 to %c32 step %c8 iter_args(%arg4 = %1) -> (tensor<?x1x6x8xf32>) {
    %4 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg3)
    %5 = tensor.insert_slice %2 into %arg4[%4,0, 0, 0] [1, 1, 6, 8] [1, 1, 1, 1] :
      tensor<1x6x8xf32> into tensor<?x1x6x8xf32>
    scf.yield %5 : tensor<?x1x6x8xf32>
  }
  return %3 : tensor<?x1x6x8xf32>
}

// -----

// CHECK-LABEL: func @write_to_select_op_source
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>
func @write_to_select_op_source(
    %t1 : tensor<?xf32> {linalg.inplaceable = true},
    %t2 : tensor<?xf32> {linalg.inplaceable = true},
    %c : i1)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[t1]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]]
  %w = tensor.insert %cst into %t1[%idx] : tensor<?xf32>
  // CHECK: %[[select:.*]] = arith.select %{{.*}}, %[[t1]], %[[t2]]
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>
  // CHECK: return %[[select]], %[[alloc]]
  return %s, %w : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_after_select_read_one
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>
func @write_after_select_read_one(
    %t1 : tensor<?xf32> {linalg.inplaceable = true},
    %t2 : tensor<?xf32> {linalg.inplaceable = true},
    %c : i1)
  -> (f32, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index

  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK: memref.copy %[[t1]], %[[alloc]]
  // CHECK: %[[select:.*]] = arith.select %{{.*}}, %[[casted]], %[[t2]]
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>

  // CHECK: memref.store %{{.*}}, %[[select]]
  %w = tensor.insert %cst into %s[%idx] : tensor<?xf32>

  // CHECK: %[[f:.*]] = memref.load %[[t1]]
  %f = tensor.extract %t1[%idx] : tensor<?xf32>

  // CHECK: return %[[f]], %[[select]]
  return %f, %w : f32, tensor<?xf32>
}

// -----

// A regression test to make sure that we handle rank-reducing extract_slice
// correctly.

// CHECK-LABEL: func @rank_reducing
func @rank_reducing(
    %i: index, %j: index,
    %arg0: tensor<8x18x32xf32>)
      -> tensor<?x1x6x8xf32> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = linalg.init_tensor [4, 1, 6, 8] : tensor<4x1x6x8xf32>
  %1 = tensor.cast %0 : tensor<4x1x6x8xf32> to tensor<?x1x6x8xf32>
  %2 = linalg.init_tensor [1, 6, 8] : tensor<1x6x8xf32>
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

// Note: This bufferization is inefficient, but it bufferizes correctly.

// CHECK-LABEL: func @scf_execute_region_yield_non_equivalent(
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}})
//       CHECK:   %[[clone:.*]] = bufferization.clone %[[alloc]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   %[[r:.*]] = memref.load %[[clone]][%{{.*}}]
//       CHECK:   memref.dealloc %[[clone]]
//       CHECK:   return %[[r]]
func @scf_execute_region_yield_non_equivalent(%i: index, %j: index) -> f32 {
  %r = scf.execute_region -> (tensor<?xf32>) {
    %t2 = linalg.init_tensor [%i] : tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  %f = tensor.extract %r[%j] : tensor<?xf32>
  return %f : f32
}

// -----

// Note: This bufferizes to inefficient code, but bufferization should not see
// such IR in the first place. The iter_arg would canonicalize away. This test
// case is just to ensure that the bufferization generates correct code.

// CHECK-LABEL: func @scf_for_yield_non_equivalent(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}})
//       CHECK:   %[[casted:.*]] = memref.cast %[[alloc]]
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%[[iter:.*]] = %[[casted]])
//       CHECK:     memref.dealloc %[[iter]]
//       CHECK:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[t]], %[[alloc2]]
//       CHECK:     %[[casted2:.*]] = memref.cast %[[alloc2]]
//       CHECK:     scf.yield %[[casted2]]
//       CHECK:   return %[[for]]
func @scf_for_yield_non_equivalent(
    %t: tensor<?xf32>, %lb : index, %ub : index, %step : index) -> tensor<?xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%a = %t) -> tensor<?xf32> {
    scf.yield %t : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// Note: This bufferizes to inefficient code, but bufferization should not see
// such IR in the first place. The iter_arg would canonicalize away. This test
// case is just to ensure that the bufferization generates correct code.

// CHECK-LABEL: func @scf_for_yield_allocation(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[cloned:.*]] = bufferization.clone %[[t]]
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%[[iter:.*]] = %[[cloned]])
// This alloc is for the linalg.init_tensor.
//       CHECK:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.dealloc %[[iter]]
// This alloc is for the scf.yield.
//       CHECK:     %[[alloc3:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[alloc2]], %[[alloc3]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     %[[casted3:.*]] = memref.cast %[[alloc3]]
//       CHECK:     scf.yield %[[casted3]]
//       CHECK:   return %[[for]]
func @scf_for_yield_allocation(%t: tensor<?xf32>, %lb : index, %ub : index,
                               %step : index) -> tensor<?xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%a = %t) -> tensor<?xf32> {
    %t2 = linalg.init_tensor [%i] : tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// TODO: The scf.yield could bufferize to 1 alloc and 2 copies (instead of
// 2 allocs and 2 copies).

// CHECK-LABEL: func @scf_for_swapping_yields(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32, #{{.*}}>, %[[B:.*]]: memref<?xf32, #{{.*}}>

func @scf_for_swapping_yields(
    %A : tensor<?xf32>, %B : tensor<?xf32> {linalg.inplaceable = true},
    %C : tensor<4xf32>, %lb : index, %ub : index, %step : index)
  -> (f32, f32)
{
//   CHECK-DAG:   %[[clone1:.*]] = bufferization.clone %[[A]]
//   CHECK-DAG:   %[[clone2:.*]] = bufferization.clone %[[B]]
//       CHECK:   %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[iter1:.*]] = %[[clone1]], %[[iter2:.*]] = %[[clone2]])
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
//       CHECK:     %[[sv1:.*]] = memref.subview %[[iter1]]
//       CHECK:     memref.copy %{{.*}}, %[[sv1]]
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
//       CHECK:     %[[sv2:.*]] = memref.subview %[[iter2]]
//       CHECK:     memref.copy %{{.*}}, %[[sv2]]
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

//       CHECK:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[iter2]], %[[alloc2]]
//       CHECK:     memref.dealloc %[[iter2]]
//       CHECK:     %[[alloc1:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[iter1]], %[[alloc1]]
//       CHECK:     memref.dealloc %[[iter1]]
//       CHECK:     %[[casted1:.*]] = memref.cast %[[alloc1]]
//       CHECK:     %[[casted2:.*]] = memref.cast %[[alloc2]]
//       CHECK:     scf.yield %[[casted2]], %[[casted1]]
    // Yield tensors in different order.
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }

//       CHECK:     %[[r0:.*]] = memref.load %[[for]]#0
//       CHECK:     memref.dealloc %[[for]]#0
//       CHECK:     %[[r1:.*]] = memref.load %[[for]]#1
//       CHECK:     memref.dealloc %[[for]]#1
  %f0 = tensor.extract %r0#0[%step] : tensor<?xf32>
  %f1 = tensor.extract %r0#1[%step] : tensor<?xf32>
//       CHECK:     return %[[r0]], %[[r1]]
  return %f0, %f1: f32, f32
}

