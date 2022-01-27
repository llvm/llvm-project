// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize=allow-return-memref -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-memref test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-memref test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-memref test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-memref fully-dynamic-layout-maps=0" -split-input-file | FileCheck %s --check-prefix=CHECK-NO-LAYOUT-MAP

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
  //     CHECK: linalg.fill(%[[F0]], %[[A]]) : f32, memref<?xf32, #[[$map_1d_dyn]]>
  %r = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

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
  //     CHECK: linalg.fill(%[[F0]], %[[ALLOC]]) : f32, memref<?xf32>
  %r = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //     CHECK:  dealloc %[[ALLOC]] : memref<?xf32>
  //     CHECK:  return %[[ALLOC]] : memref<?xf32>
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
  //       CHECK: linalg.fill({{.*}}, %[[ALLOC]]
  %f = linalg.fill(%f0, %A) : f32, tensor<?x?xf32> -> tensor<?x?xf32>

  /// The second op has no interfering reads and can reuse.
  //   CHECK-NOT: alloc
  //       CHECK: linalg.matmul ins(%[[ALLOC]], %[[ALLOC]]{{.*}}) outs(%[[A]]
  %r = linalg.matmul  ins(%f, %f: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>

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
  //      CHECK: %[[REALLOC3:.*]] = memref.alloc
  //      CHECK: %[[REALLOC2:.*]] = memref.alloc
  //      CHECK: %[[REALLOC1:.*]] = memref.alloc

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
  //      CHECK: linalg.fill({{.*}}, %[[A]]
  %r1 = linalg.fill(%f0, %r0) : f32, tensor<?xf32> -> tensor<?xf32>

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

  //      CHECK: linalg.fill({{.*}}, %[[A]]
  %r0 = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

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
  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
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

  //     CHECK:   memref.dealloc %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:   return %[[CASTED]] : memref<?xf32, #[[$map_1d_dyn]]>
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

  //     CHECK:  memref.dealloc %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:  return %[[CASTED]] : memref<?xf32, #[[$map_1d_dyn]]>
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Cross function boundary cases.
//===----------------------------------------------------------------------===//

//      CHECK: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK: memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>
//      CHECK: func private @some_external_func(memref<4xi32, #[[$DYN_1D_MAP]]>)
func private @some_external_func(tensor<4xi32>)

//      CHECK: func @main()
func @main() {
//      CHECK:   %[[A:.*]] = memref.get_global @__constant_4xi32 : memref<4xi32>
  %A = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

//      CHECK:   %[[alloc:.*]] = memref.alloc
//      CHECK:   %[[B:.*]] = memref.cast %[[alloc]] : memref<4xi32> to memref<4xi32, #[[$DYN_1D_MAP]]>
//      CHECK:   memref.copy %[[A]], %[[alloc]]
//      CHECK:   call @some_external_func(%[[B]]) : (memref<4xi32, #[[$DYN_1D_MAP]]>) -> ()
  call @some_external_func(%A) : (tensor<4xi32>) -> ()

  return
}

// -----

//      CHECK: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK: memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>
//      CHECK: func private @some_external_func_within_scf_execute(memref<4xi32, #[[$DYN_1D_MAP]]>)
func private @some_external_func_within_scf_execute(tensor<4xi32>)

//      CHECK: func @main()
func @main() {
//      CHECK:   %[[A:.*]] = memref.get_global @__constant_4xi32 : memref<4xi32>
  %A = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

//      CHECK:   %[[alloc:.*]] = memref.alloc
//      CHECK:   %[[B:.*]] = memref.cast %[[alloc]] : memref<4xi32> to memref<4xi32, #[[$DYN_1D_MAP]]>
//      CHECK:   memref.copy %[[A]], %[[alloc]]
//      CHECK:   call @some_external_func_within_scf_execute(%[[B]]) : (memref<4xi32, #[[$DYN_1D_MAP]]>) -> ()
  scf.execute_region {
    call @some_external_func_within_scf_execute(%A) : (tensor<4xi32>) -> ()
    scf.yield
  }

  return
}

// -----

// CHECK-LABEL: func @execute_region_test(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @execute_region_test(%t1 : tensor<?xf32> {linalg.inplaceable = "true"})
    -> (f32, tensor<?xf32>, f32)
{
  %f1 = arith.constant 0.0 : f32
  %f2 = arith.constant 1.0 : f32
  %idx = arith.constant 7 : index

  // scf.execute_region is canonicalized away after bufferization. So just the
  // memref.store is left over.

  // CHECK: memref.store %{{.*}}, %[[m1]][%{{.*}}]
  %0, %1, %2 = scf.execute_region -> (f32, tensor<?xf32>, f32) {
    %t2 = tensor.insert %f2 into %t1[%idx] : tensor<?xf32>
    scf.yield %f1, %t2, %f2 : f32, tensor<?xf32>, f32
  }

  // CHECK: return %{{.*}}, %{{.*}} : f32, f32
  return %0, %1, %2 : f32, tensor<?xf32>, f32
}

// -----

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

//      CHECK: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK:  func private @some_external_func(memref<?xf32, #[[$DYN_1D_MAP]]>)
func private @some_external_func(tensor<?xf32>)

//      CHECK:  func @scf_for_with_tensor_insert_slice(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$DYN_1D_MAP]]>
func @scf_for_with_tensor_insert_slice(
    %A : tensor<?xf32>, %B : tensor<?xf32>, %C : tensor<4xf32>,
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK-NEXT: scf.for
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // CHECK-NEXT:   %[[SVA:.*]] = memref.subview %[[A]]
    // CHECK-NEXT:   memref.copy %[[C]], %[[SVA]] : memref<4xf32, #[[$DYN_1D_MAP]]> to memref<4xf32, #[[$DYN_1D_MAP]]>
    %ttA = tensor.insert_slice %C into %tA[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NEXT:   %[[SVB:.*]] = memref.subview %[[B]]
    // CHECK-NEXT:   memref.copy %[[C]], %[[SVB]] : memref<4xf32, #[[$DYN_1D_MAP]]> to memref<4xf32, #[[$DYN_1D_MAP]]>
    %ttB = tensor.insert_slice %C into %tB[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // scf.yield is empty and is elided
    //  CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  // Swaparoo requires bufferizing the whole function to figure out who's who.
  return %r0#1, %r0#0: tensor<?xf32>, tensor<?xf32>
}

//      CHECK:  func @bar(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$DYN_1D_MAP]]>
func @bar(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32> {linalg.inplaceable = true},
    %C : tensor<4xf32> {linalg.inplaceable = true},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
//      CHECK:   call @scf_for_with_tensor_insert_slice(%[[A]], %[[B]], %[[C]]
  %r0:2 = call @scf_for_with_tensor_insert_slice(%A, %B, %C, %lb, %ub, %step) :
      (tensor<?xf32>, tensor<?xf32>, tensor<4xf32>, index, index, index)
        -> (tensor<?xf32>, tensor<?xf32>)

  // %r0#0 requires a copy because we have no idea what the function is doing.
//      CHECK:   %[[alloc:.*]] = memref.alloc
//      CHECK:   %[[casted:.*]] = memref.cast %[[alloc]]
//      CHECK:   memref.copy %[[B]], %[[alloc]]
// CHECK-NEXT:   call @some_external_func(%[[casted]]) : (memref<?xf32, #[[$DYN_1D_MAP]]>) -> ()
  call @some_external_func(%r0#0) : (tensor<?xf32>) -> ()

//      CHECK:   return
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

//  CHECK-DAG: #[[$DYN_0D_MAP:.*]] = affine_map<()[s0] -> (s0)>
//  CHECK-DAG: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK:  func @init_and_dot(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<64xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<64xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<f32, #[[$DYN_0D_MAP]]>
func @init_and_dot(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT:   %[[C0:.*]] = arith.constant 0{{.*}} : f32
  %v0 = arith.constant 0.0 : f32

  // CHECK-NEXT:   linalg.fill(%[[C0]], %[[C]]) : f32, memref<f32, #[[$DYN_0D_MAP]]>
  %d = linalg.fill(%v0, %c) : f32, tensor<f32> -> tensor<f32>

  // CHECK-NEXT:   linalg.dot ins(%[[A]], %[[B]] : memref<64xf32, #[[$DYN_1D_MAP]]>, memref<64xf32, #[[$DYN_1D_MAP]]>) outs(%[[C]] : memref<f32, #[[$DYN_0D_MAP]]>)
  %e = linalg.dot ins(%a, %b : tensor<64xf32>,tensor<64xf32>)
    outs(%d: tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   return
  return %e : tensor<f32>
}

//      CHECK:  func @main()
func @main() {
  //  CHECK-DAG:   %[[C0:.*]] = arith.constant 0{{.*}} : f32
  //  CHECK-DAG:   %[[C1:.*]] = arith.constant 1{{.*}} : f32
  //  CHECK-DAG:   %[[C2:.*]] = arith.constant 2{{.*}} : f32
  %v0 = arith.constant 0.0 : f32
  %v1 = arith.constant 1.0 : f32
  %v2 = arith.constant 2.0 : f32

  // CHECK-NEXT:   %[[A:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[B:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[C:.*]] = memref.alloc() {alignment = 128 : i64} : memref<f32>
  // CHECK-NEXT:   %[[cA:.*]] = memref.cast %[[A]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  // CHECK-NEXT:   %[[cB:.*]] = memref.cast %[[B]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  // CHECK-NEXT:   %[[cC:.*]] = memref.cast %[[C]] : memref<f32> to memref<f32, #[[$DYN_0D_MAP]]>
  %A = linalg.init_tensor [64] : tensor<64xf32>
  %B = linalg.init_tensor [64] : tensor<64xf32>
  %C = linalg.init_tensor [] : tensor<f32>

  // CHECK-NEXT:   linalg.fill(%[[C1]], %[[A]]) : f32, memref<64xf32>
  // CHECK-NEXT:   linalg.fill(%[[C2]], %[[B]]) : f32, memref<64xf32>
  // CHECK-NEXT:   linalg.fill(%[[C0]], %[[C]]) : f32, memref<f32>
  %AA = linalg.fill(%v1, %A) : f32, tensor<64xf32> -> tensor<64xf32>
  %BB = linalg.fill(%v2, %B) : f32, tensor<64xf32> -> tensor<64xf32>
  %CC = linalg.fill(%v0, %C) : f32, tensor<f32> -> tensor<f32>

  // CHECK-NEXT:   call @init_and_dot(%[[cA]], %[[cB]], %[[cC]])
  %res = call @init_and_dot(%AA, %BB, %CC) :
    (tensor<64xf32>, tensor<64xf32>, tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   %[[dC:.*]] = memref.cast %[[C]] : memref<f32> to memref<*xf32>
  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

  // CHECK-NEXT:   call @print_memref_f32(%[[dC]]) : (memref<*xf32>) -> ()
  call @print_memref_f32(%res2) : (tensor<*xf32>) -> ()

  // CHECK-DAG:   memref.dealloc %[[A]] : memref<64xf32>
  // CHECK-DAG:   memref.dealloc %[[B]] : memref<64xf32>
  // CHECK-DAG:   memref.dealloc %[[C]] : memref<f32>
  // CHECK-NEXT:   return
  return
}

//     CHECK:   func private @print_memref_f32(memref<*xf32>)
func private @print_memref_f32(tensor<*xf32>)

// -----

func private @some_use(memref<?xf32>)

#TILE_MAP = affine_map<(d0)[s0] -> (3, -d0 + s0)>

//  CHECK-DAG: #[[$DYN_0D_MAP:.*]] = affine_map<()[s0] -> (s0)>
//  CHECK-DAG: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//  CHECK-DAG: #[[$TILE_MAP:.*]] = affine_map<(d0)[s0] -> (3, -d0 + s0)>

//      CHECK:  func @tiled_dot(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[c:[a-zA-Z0-9]*]]: memref<f32, #[[$DYN_0D_MAP]]>
func @tiled_dot(
    %A: tensor<?xf32> {linalg.inplaceable = false},
    %B: tensor<?xf32> {linalg.inplaceable = false},
    %c: tensor<f32> {linalg.inplaceable = true},
    %effecting: memref<?xf32>)
  -> tensor<f32>
{
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, #[[$DYN_1D_MAP:.*]]>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: linalg.tiled_loop {{.*}} to (%[[M]]) {{.*}} %[[A]]{{.*}}%[[B]]{{.*}}outs{{.*}}%[[c]]
  // CHECK-NOT: copy
  %1 = linalg.tiled_loop (%arg3) = (%c0) to (%0) step (%c3)
       ins (%arg4 = %A: tensor<?xf32>, %use = %effecting : memref<?xf32>, %arg5 = %B: tensor<?xf32>)
      outs (%arg6 = %c: tensor<f32>)
      iterators["reduction"]
  {
    // CHECK-NOT:   alloc

    %2 = tensor.dim %arg4, %c0 : tensor<?xf32>
    %3 = affine.min #TILE_MAP(%arg3)[%2]

    //     CHECK:   %[[SV_A:.*]] = memref.subview {{.*}}
    %4 = tensor.extract_slice %arg4[%arg3] [%3] [1] : tensor<?xf32> to tensor<?xf32>
    %5 = tensor.dim %arg5, %c0 : tensor<?xf32>
    %6 = affine.min #TILE_MAP(%arg3)[%5]

    //     CHECK:   %[[SV_B:.*]] = memref.subview {{.*}}
    %7 = tensor.extract_slice %arg5[%arg3] [%6] [1] : tensor<?xf32> to tensor<?xf32>

    //     CHECK:   linalg.dot ins(%[[SV_A]], %[[SV_B]] : memref<?xf32, #[[$DYN_1D_MAP:.*]]>, memref<?xf32, #[[$DYN_1D_MAP:.*]]>) outs(%{{.*}} : memref<f32, #[[$DYN_0D_MAP]]>)
    %8 = linalg.dot ins(%4, %7 : tensor<?xf32>, tensor<?xf32>) outs(%arg6 : tensor<f32>) -> tensor<f32>

    //     CHECK:   call @some_use(%{{.*}}) : (memref<?xf32>) -> ()
    call @some_use(%use) : (memref<?xf32>) -> ()

    linalg.yield %8 : tensor<f32>
    //     CHECK:   linalg.yield
    // CHECK-NOT:   tensor
  }

  //     CHECK: return
  // CHECK-NOT: tensor
  return %1 : tensor<f32>
}

// -----

#TILE_MAP = affine_map<(d0)[s0] -> (3, -d0 + s0)>

//  CHECK-DAG: #[[$DYN_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK:  func @tiled_fill(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_MAP]]>
func @tiled_fill(%A: tensor<?xf32> {linalg.inplaceable = true}) -> tensor<?xf32> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, #[[$DYN_MAP:.*]]>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: linalg.tiled_loop {{.*}} to (%[[M]]) {{.*}} outs{{.*}}%[[A]]
  %1 = linalg.tiled_loop (%arg3) = (%c0) to (%0) step (%c3)
      outs (%arg1 = %A: tensor<?xf32>)
      iterators["parallel"]
  {
    // CHECK-NOT:   alloc

    %2 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %3 = affine.min #TILE_MAP(%arg3)[%2]

    //     CHECK:   %[[SV_A:.*]] = memref.subview {{.*}}
    %4 = tensor.extract_slice %arg1[%arg3] [%3] [1] : tensor<?xf32> to tensor<?xf32>

    //     CHECK:   linalg.fill(%{{.*}}, %[[SV_A]]) : f32, memref<?xf32, #[[$DYN_MAP:.*]]>
    %5 = linalg.fill(%f0, %4) : f32, tensor<?xf32> -> tensor<?xf32>
    %6 = tensor.insert_slice %5 into %arg1[%arg3] [%3] [1] : tensor<?xf32> into tensor<?xf32>

    linalg.yield %6 : tensor<?xf32>
    //     CHECK:   linalg.yield
    // CHECK-NOT:   tensor
  }

  //     CHECK: return
  // CHECK-NOT: tensor
  return %1 : tensor<?xf32>
}

// -----

//      CHECK:  func @tiled_loop_yield_out_of_place(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #{{.*}}>,
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #{{.*}}>
func @tiled_loop_yield_out_of_place(
    %A: tensor<?xf32> {linalg.inplaceable = true},
    %B: tensor<?xf32> {linalg.inplaceable = true})
  -> tensor<?xf32>
{
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, #[[$DYN_MAP:.*]]>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: linalg.tiled_loop {{.*}} to (%[[M]]) {{.*}} outs{{.*}}%[[A]]
  %1 = linalg.tiled_loop (%arg3) = (%c0) to (%0) step (%c3)
      outs (%arg1 = %A: tensor<?xf32>)
      iterators["parallel"]
  {
    // CHECK-NOT:   alloc
    //     CHECK:   memref.copy %[[B]], %[[A]]
    linalg.yield %B : tensor<?xf32>
    //     CHECK:   linalg.yield
    // CHECK-NOT:   tensor
  }

  //     CHECK: return
  // CHECK-NOT: tensor
  return %1 : tensor<?xf32>
}

// -----

// CHECK: #[[$DYNAMIC:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK: func private @external_func(memref<?xf32, #[[$DYNAMIC]]>)
func private @external_func(tensor<?xf32>)

//      CHECK: func @callee(
// CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[B:[0-9a-zA-Z]*]]: memref<?xf32, #[[$DYNAMIC]]>
// CHECK-SAME:   %[[C:[0-9a-zA-Z]*]]: memref<?xf32, #[[$DYNAMIC]]>
func @callee(%A : tensor<?xf32> {linalg.buffer_layout = affine_map<(i)[s0, s1] -> (i)>},
             %B : tensor<?xf32>,
             %C : tensor<?xf32>) {
// CHECK-NEXT: %[[CASTED:.*]] = memref.cast %[[A]] : memref<?xf32> to memref<?xf32, #[[$DYNAMIC]]>
// CHECK-NEXT: call @external_func(%[[CASTED]]) : (memref<?xf32, #[[$DYNAMIC]]>) -> ()
  call @external_func(%A) : (tensor<?xf32>) -> ()

// CHECK-NEXT: call @external_func(%[[B]]) : (memref<?xf32, #[[$DYNAMIC]]>) -> ()
  call @external_func(%B) : (tensor<?xf32>) -> ()

// CHECK-NEXT: call @external_func(%[[C]]) : (memref<?xf32, #[[$DYNAMIC]]>) -> ()
  call @external_func(%C) : (tensor<?xf32>) -> ()

  return
}

//      CHECK: func @entry(
// CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[B:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[C:[0-9a-zA-Z]*]]: memref<?xf32, #[[$DYNAMIC]]>
func @entry(%A : tensor<?xf32> {linalg.buffer_layout = affine_map<(i)[s0, s1] -> (i)>, linalg.inplaceable = false},
            %B : tensor<?xf32> {linalg.buffer_layout = affine_map<(i)[s0, s1] -> (i)>, linalg.inplaceable = false},
            %C : tensor<?xf32> {linalg.inplaceable = false}) {
// Note: `callee` does not write to its bbArg directly, but `external_func`
// does. Inside `callee`, the writes via `external_func` do not cause a
// conflict. However, inside `entry`, the writes do cause a conflict because
// %A, %B and %C are not inplaceable. This test case shows that this kind of
// conflict detection has a "transitive" nature.
//      CHECK: %[[ALLOC_C:.*]] = memref.alloc
//      CHECK: %[[CASTED_C:.*]] = memref.cast %[[ALLOC_C]]
//      CHECK: %[[ALLOC_B:.*]] = memref.alloc
//      CHECK: %[[CASTED_B:.*]] = memref.cast %[[ALLOC_B]]
//      CHECK: %[[ALLOC_A:.*]] = memref.alloc
//      CHECK: memref.copy %[[A]], %[[ALLOC_A]]
//      CHECK: memref.copy %[[B]], %[[ALLOC_B]]
//      CHECK: memref.copy %[[C]], %[[ALLOC_C]]
//      CHECK: %[[CASTED_A:.*]] = memref.cast %[[ALLOC_A]]
// CHECK-NEXT: call @callee(%[[CASTED_A]], %[[CASTED_B]], %[[CASTED_C]])
  call @callee(%A, %B, %C) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> ()
  return
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
      // CHECK: %[[T:.*]] = memref.subview %[[C]][%[[I]], %[[J]]] [8, 16] [1, 1]
      %4 = tensor.extract_slice %C[%arg3, %arg5] [8, 16] [1, 1] :
        tensor<128x192xf32> to tensor<8x16xf32>

      // linalg.fill is inplace.
      // CHECK: linalg.fill(%{{.*}}, %[[ALLOC]]) : f32, memref<8x16xf32>
      %5 = linalg.fill(%cst, %4) : f32, tensor<8x16xf32> -> tensor<8x16xf32>

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
  %FA = linalg.fill(%f0, %ssA) : f32, tensor<4x4xf32> -> tensor<4x4xf32>
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
  // CHECK: %[[r:.*]] = select %[[cond]], %[[A]], %[[B]]
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

// CHECK-LABEL: func @inner_func(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func @inner_func(%t: tensor<?xf32>) -> tensor<?xf32> {
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @equivalent_func_arg(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func @equivalent_func_arg(%t0: tensor<?xf32> {linalg.inplaceable = true},
                          %c0: index, %c10: index, %c1: index) -> tensor<?xf32> {
  // CHECK-NOT: copy
  %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t1 = %t0) -> (tensor<?xf32>) {
    // CHECK: call @inner_func(%[[arg0]])
    %3 = call @inner_func(%t1) : (tensor<?xf32>) -> tensor<?xf32>
    scf.yield %3 : tensor<?xf32>
  }
  return %1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @inner_func_2(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func @inner_func_2(%t: tensor<?xf32>) -> tensor<?xf32> {
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @equivalent_func_arg_2(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func @equivalent_func_arg_2(%t0: tensor<?xf32> {linalg.inplaceable = true},
                            %c0: index, %c10: index, %c1: index) -> tensor<?xf32> {
  %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t1 = %t0) -> (tensor<?xf32>) {
    // CHECK: %[[alloc:.*]] = memref.alloc
    // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
    // CHECK: memref.copy %[[arg0]], %[[alloc]]
    // CHECK: call @inner_func_2(%[[casted]])
    %3 = call @inner_func_2(%t1) : (tensor<?xf32>) -> tensor<?xf32>
    scf.yield %t1 : tensor<?xf32>
  }
  return %1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @inner_func(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func @inner_func(%t: tensor<?xf32>) -> (tensor<?xf32>, f32) {
  // CHECK-NOT: copy
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  // CHECK: %[[load:.*]] = memref.load %[[arg0]]
  %1 = tensor.extract %0[%c1] : tensor<?xf32>
  // CHECK: return %[[load]] : f32
  return %0, %1 : tensor<?xf32>, f32
}

// CHECK-LABEL: func @call_func_with_non_tensor_return(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func @call_func_with_non_tensor_return(
    %t0: tensor<?xf32> {linalg.inplaceable = true}) -> (f32, tensor<?xf32>) {
  // CHECK-NOT: copy
  // CHECK: %[[call:.*]] = call @inner_func(%[[arg0]])
  %0, %1 = call @inner_func(%t0) : (tensor<?xf32>) -> (tensor<?xf32>, f32)
  // CHECK: return %[[call]] : f32
  return %1, %0 : f32, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @func_without_tensor_args
func @func_without_tensor_args(%v : vector<10xf32>) -> () {
  // CHECK: %[[alloc:.*]] = memref.alloc()
  %0 = linalg.init_tensor[10] : tensor<10xf32>

  %c0 = arith.constant 0 : index
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  %1 = vector.transfer_write %v, %0[%c0] : vector<10xf32>, tensor<10xf32>

  %cst = arith.constant 0.0 : f32
  // CHECK: vector.transfer_read %[[alloc]]
  %r = vector.transfer_read %1[%c0], %cst : tensor<10xf32>, vector<11xf32>

  vector.print %r : vector<11xf32>
  return
}

// -----

// CHECK-LABEL: func private @private_func
func private @private_func(tensor<?xf32>) -> ()

// CHECK-LABEL: func @empty_func()
func @empty_func() -> () {
  return
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
    %t2: tensor<?xf32> {linalg.inplaceable = false},
    %t3: tensor<?x?xf32> {linalg.inplaceable = false},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[t1]] : {{.*}})
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

// CHECK-LABEL: func @linalg_op_bufferizes_out_of_place_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>, %[[t3:.*]]: memref<?x?xf32, #{{.*}}>
func @linalg_op_bufferizes_out_of_place_with_input(
    %t1: tensor<?x?xf32> {linalg.inplaceable = false},
    %t2: tensor<?xf32> {linalg.inplaceable = false},
    %t3: tensor<?x?xf32> {linalg.inplaceable = false},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[t1]], %[[alloc]]
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[alloc]] : {{.*}})
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
  // CHECK: return %[[alloc]]
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @linalg_op_output_cannot_alias_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>, %[[t3:.*]]: memref<?x?xf32, #{{.*}}>
func @linalg_op_output_cannot_alias_with_input(
    %t1: tensor<?x?xf32> {linalg.inplaceable = true},
    %t2: tensor<?xf32> {linalg.inplaceable = false},
    %t3: tensor<?x?xf32> {linalg.inplaceable = true},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[t3]] : {{.*}})
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
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
  // CHECK: %[[select:.*]] = select %{{.*}}, %[[t1]], %[[t2]]
  %s = std.select %c, %t1, %t2 : tensor<?xf32>
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
  // CHECK: %[[select:.*]] = select %{{.*}}, %[[casted]], %[[t2]]
  %s = std.select %c, %t1, %t2 : tensor<?xf32>

  // CHECK: memref.store %{{.*}}, %[[select]]
  %w = tensor.insert %cst into %s[%idx] : tensor<?xf32>

  // CHECK: %[[f:.*]] = memref.load %[[t1]]
  %f = tensor.extract %t1[%idx] : tensor<?xf32>

  // CHECK: return %[[f]], %[[select]]
  return %f, %w : f32, tensor<?xf32>
}
