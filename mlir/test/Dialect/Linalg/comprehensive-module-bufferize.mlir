// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize=allow-return-memref -split-input-file | FileCheck %s

// CHECK-LABEL: func @transfer_read(%{{.*}}: memref<?xf32, #map>) -> vector<4xf32> {
func @transfer_read(%A : tensor<?xf32>) -> (vector<4xf32>) {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32

//       CHECK: %[[RES:.*]] = vector.transfer_read {{.*}} : memref<?xf32, #{{.*}}>, vector<4xf32>
  %0 = vector.transfer_read %A[%c0], %f0 : tensor<?xf32>, vector<4xf32>

//       CHECK: return %[[RES]] : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @fill_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
func @fill_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}) -> tensor<?xf32> {
  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

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
func @tensor_extract(%A : tensor<?xf32>) -> (f32) {
  %c0 = constant 0 : index

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
func @not_inplace(%A : tensor<?xf32>) -> tensor<?xf32> {
  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

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
func @not_inplace(%A : tensor<?x?xf32> {linalg.inplaceable = true}) -> tensor<?x?xf32> {
  %f0 = constant 0.0 : f32

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
  %c0 = constant 0 : index

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
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  /// Cross-op multiple uses of %A, the first vector.transfer which has interfering reads must alloc.
  //      CHECK: %[[ALLOC:.*]] = memref.alloc
  //      CHECK: linalg.copy({{.*}}, %[[ALLOC]])
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
func @insert_slice_fun(%A0 : tensor<?xf32>,
                       %A1 : tensor<?xf32> {linalg.inplaceable = true},
                       %t0 : tensor<4xf32>,
                       %t1 : tensor<4xf32> {linalg.inplaceable = true})
  ->  (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
{
  // Hoisted allocs.
  //      CHECK: %[[REALLOC_A1:.*]] = memref.alloc
  //      CHECK: %[[REALLOC_A0_2:.*]] = memref.alloc
  //      CHECK: %[[REALLOC_A0:.*]] = memref.alloc

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: linalg.copy(%[[A0]], %[[REALLOC_A0]]
  //      CHECK: %[[SV_A0:.*]] = memref.subview %[[REALLOC_A0]]
  //      CHECK: linalg.copy(%[[t0]], %[[SV_A0]])
  %r0 = tensor.insert_slice %t0 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: linalg.copy(%[[A0]]
  //      CHECK: %[[SV_A0_2:.*]] = memref.subview %[[REALLOC_A0_2]]
  //      CHECK: linalg.copy(%[[t1]], %[[SV_A0_2]])
  %r1 = tensor.insert_slice %t1 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Still alloc the large tensor because %A1 is read after. Copy the tensor.extract_slice.
  //      CHECK: linalg.copy(%[[A1]]
  //      CHECK: %[[SV_A1:.*]] = memref.subview %[[REALLOC_A1]]
  //      CHECK: linalg.copy(%[[t0]], %[[SV_A1]])
  %r2 = tensor.insert_slice %t0 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Do not realloc the large tensor. Copy the tensor.extract_slice.
  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A1_2:.*]] = memref.subview %[[A1]]
  //      CHECK: linalg.copy(%[[t1]], %[[SV_A1_2]])
  %r3 = tensor.insert_slice %t1 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return %[[REALLOC_A0]], %[[REALLOC_A0_2]], %[[REALLOC_A1]] :
  // CHECK-SAME:   memref<?xf32>, memref<?xf32>, memref<?xf32>
  return %r0, %r1, %r2, %r3: tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func @insert_slice_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  %f0 = constant 0.0 : f32

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A:.*]] = memref.subview %[[A]]
  //      CHECK: linalg.copy(%[[t]], %[[SV_A]])
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
func @insert_slice_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  %f0 = constant 0.0 : f32

  //      CHECK: linalg.fill({{.*}}, %[[A]]
  %r0 = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A:.*]] = memref.subview %[[A]]
  /// Overwrite A inplace by copying into the subview.
  //      CHECK: linalg.copy(%[[t]], %[[SV_A]])
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
func @insert_slice_fun_not_inplace(%A : tensor<?xf32>, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) {alignment = 128 : i64} : memref<?xf32>
  //      CHECK: linalg.copy(%[[A]], %[[ALLOC]]) : memref<?xf32{{.*}}, memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32>
  //      CHECK: linalg.copy(%[[t]], %[[SV]]) : memref<4xf32, #map>, memref<4xf32>
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
func @scf_for_yield_only(%A : tensor<?xf32>,
                         %B : tensor<?xf32> {linalg.inplaceable = true},
                         %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   linalg.copy(%[[A]], %[[ALLOC_FOR_A]])

  // The first scf.for remains but just turns into dead code.
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  // The second scf.for remains but just turns into dead code.
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //     CHECK:   memref.dealloc %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:   return %[[ALLOC_FOR_A]] : memref<?xf32>
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func @scf_for_with_tensor.insert_slice(
   %A : tensor<?xf32>,
   %B : tensor<?xf32> {linalg.inplaceable = true},
   %C : tensor<4xf32>,
   %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   linalg.copy(%[[A]], %[[ALLOC_FOR_A]])

  //     CHECK: %[[svA:.*]] = memref.subview %[[ALLOC_FOR_A]][0] [4] [1]
  //     CHECK: %[[svB:.*]] = memref.subview %[[B]][0] [4] [1]

  //     CHECK:   scf.for {{.*}}
  // CHECK-NOT: iter_args
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // %ttA bufferizes to direct copy of %BUFFER_CAST_C into %svA
    //     CHECK: linalg.copy(%[[C]], %[[svA]])
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // %ttB bufferizes to direct copy of %BUFFER_CAST_C into %BUFFER_CAST_B
    //     CHECK:   linalg.copy(%[[C]], %[[svB]])
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  //     CHECK:  memref.dealloc %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:  return %[[ALLOC_FOR_A]] : memref<?xf32>
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
  %A = constant dense<[1, 2, 3, 4]> : tensor<4xi32>

//      CHECK:   %[[B:.*]] = memref.cast %[[A]] : memref<4xi32> to memref<4xi32, #[[$DYN_1D_MAP]]>
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
  %A = constant dense<[1, 2, 3, 4]> : tensor<4xi32>

//      CHECK:   %[[B:.*]] = memref.cast %[[A]] : memref<4xi32> to memref<4xi32, #[[$DYN_1D_MAP]]>
//      CHECK:   call @some_external_func_within_scf_execute(%[[B]]) : (memref<4xi32, #[[$DYN_1D_MAP]]>) -> ()
  scf.execute_region {
    call @some_external_func_within_scf_execute(%A) : (tensor<4xi32>) -> ()
    scf.yield
  }

  return
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
    // CHECK-NEXT:   linalg.copy(%[[C]], %[[SVA]]) : memref<4xf32, #[[$DYN_1D_MAP]]>, memref<4xf32, #[[$DYN_1D_MAP]]>
    %ttA = tensor.insert_slice %C into %tA[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NEXT:   %[[SVB:.*]] = memref.subview %[[B]]
    // CHECK-NEXT:   linalg.copy(%[[C]], %[[SVB]]) : memref<4xf32, #[[$DYN_1D_MAP]]>, memref<4xf32, #[[$DYN_1D_MAP]]>
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
// CHECK-NEXT:   call @scf_for_with_tensor_insert_slice(%[[A]], %[[B]], %[[C]]
  %r0:2 = call @scf_for_with_tensor_insert_slice(%A, %B, %C, %lb, %ub, %step) :
      (tensor<?xf32>, tensor<?xf32>, tensor<4xf32>, index, index, index)
        -> (tensor<?xf32>, tensor<?xf32>)

  // %r0#0 is actually %B after inplaceable results are swapped in the callee.
// CHECK-NEXT:   call @some_external_func(%[[B]]) : (memref<?xf32, #[[$DYN_1D_MAP]]>) -> ()
  call @some_external_func(%r0#0) : (tensor<?xf32>) -> ()

// CHECK-NEXT:   return
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
  // CHECK-NEXT:   %[[C0:.*]] = constant 0{{.*}} : f32
  %v0 = constant 0.0 : f32

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
  //  CHECK-DAG:   %[[C0:.*]] = constant 0{{.*}} : f32
  //  CHECK-DAG:   %[[C1:.*]] = constant 1{{.*}} : f32
  //  CHECK-DAG:   %[[C2:.*]] = constant 2{{.*}} : f32
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  // CHECK-NEXT:   %[[C:.*]] = memref.alloc() {alignment = 128 : i64} : memref<f32>
  // CHECK-NEXT:   %[[B:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[A:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
  %A = linalg.init_tensor [64] : tensor<64xf32>
  %B = linalg.init_tensor [64] : tensor<64xf32>
  %C = linalg.init_tensor [] : tensor<f32>

  // CHECK-NEXT:   linalg.fill(%[[C1]], %[[A]]) : f32, memref<64xf32>
  // CHECK-NEXT:   linalg.fill(%[[C2]], %[[B]]) : f32, memref<64xf32>
  // CHECK-NEXT:   linalg.fill(%[[C0]], %[[C]]) : f32, memref<f32>
  %AA = linalg.fill(%v1, %A) : f32, tensor<64xf32> -> tensor<64xf32>
  %BB = linalg.fill(%v2, %B) : f32, tensor<64xf32> -> tensor<64xf32>
  %CC = linalg.fill(%v0, %C) : f32, tensor<f32> -> tensor<f32>

  // CHECK-NEXT:   %[[cA:.*]] = memref.cast %[[A]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  // CHECK-NEXT:   %[[cB:.*]] = memref.cast %[[B]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  // CHECK-NEXT:   %[[cC:.*]] = memref.cast %[[C]] : memref<f32> to memref<f32, #[[$DYN_0D_MAP]]>
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
func @tiled_dot(%A: tensor<?xf32>, %B: tensor<?xf32>, %c: tensor<f32> {linalg.inplaceable = true},
                %effecting: memref<?xf32>) -> tensor<f32> {
  %c3 = constant 3 : index
  %c0 = constant 0 : index

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, #[[$DYN_1D_MAP:.*]]>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: linalg.tiled_loop {{.*}} to (%[[M]]) {{.*}} %[[A]]{{.*}}%[[B]]{{.*}}outs{{.*}}%[[c]]
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
  %c3 = constant 3 : index
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32

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
func @entry(%A : tensor<?xf32> {linalg.buffer_layout = affine_map<(i)[s0, s1] -> (i)>},
            %B : tensor<?xf32> {linalg.buffer_layout = affine_map<(i)[s0, s1] -> (i)>},
            %C : tensor<?xf32>) {
// CHECK-NEXT: %[[CASTED_B:.*]] = memref.cast %[[B]] : memref<?xf32> to memref<?xf32, #[[$DYNAMIC]]>
// CHECK-NEXT: call @callee(%[[A]], %[[CASTED_B]], %[[C]])
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
  %c0 = constant 0 : index
  %c256 = constant 256 : index
  %c32 = constant 32 : index
  %cst = constant 0.000000e+00 : f32
  %c128 = constant 128 : index
  %c192 = constant 192 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index

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
      // TODO: %4 is never read but just overwritten, this copy can be elided.
      // CHECK: linalg.copy(%[[T]], %[[ALLOC]])
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
      // CHECK: linalg.copy(%[[ALLOC]], %[[T]])
      %7 = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [8, 16] [1, 1] :
        tensor<8x16xf32> into tensor<128x192xf32>

      // CHECK: memref.dealloc %[[ALLOC]]
      scf.yield %7 : tensor<128x192xf32>
    }
    scf.yield %2 : tensor<128x192xf32>
  }
  return %0 : tensor<128x192xf32>
}
