// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize -split-input-file | FileCheck %s

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

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

/// No linalg.inplaceable flag, must allocate.
// CHECK-LABEL: func @not_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>) -> memref<?xf32> {
func @not_inplace(%A : tensor<?xf32>) -> tensor<?xf32> {
  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

  //     CHECK: %[[D0:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, #[[$map_1d_dyn]]>
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[D0]]) : memref<?xf32>
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
  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC_A0:.*]] = memref.alloc
  //      CHECK: linalg.copy(%[[A0]], %[[REALLOC_A0]]
  //      CHECK: %[[SV_A0:.*]] = memref.subview %[[REALLOC_A0]]
  //      CHECK: linalg.copy(%[[t0]], %[[SV_A0]])
  %r0 = tensor.insert_slice %t0 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC_A0_2:.*]] = memref.alloc
  //      CHECK: linalg.copy(%[[A0]]
  //      CHECK: %[[SV_A0_2:.*]] = memref.subview %[[REALLOC_A0_2]]
  //      CHECK: linalg.copy(%[[t1]], %[[SV_A0_2]])
  %r1 = tensor.insert_slice %t1 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Still alloc the large tensor because %A1 is read after. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC_A1:.*]] = memref.alloc
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
  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  //      CHECK: linalg.copy(%[[A]], %[[ALLOC]]) : memref<?xf32{{.*}}, memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32>
  //      CHECK: linalg.copy(%[[t]], %[[SV]]) : memref<4xf32, #map>, memref<4xf32>
  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //     CHECK: return %{{.*}} : memref<?xf32>
  return %r0: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @insert_slice_fun_not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func @insert_slice_fun_not_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %f0 = constant 0.0 : f32

  // tensor.insert_slice is bufferized first, %A is inplaceable so we can make this inplace
  //  CHECK-DAG: %[[SV_A:.*]] = memref.subview %[[A]][0] [4] [1] : memref<?xf32, {{.*}}> to memref<4xf32, {{.*}}>
  //  CHECK-DAG: linalg.copy(%[[t]], %[[SV_A]]) : memref<4xf32, {{.*}}>, memref<4xf32, {{.*}}>
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // fill would interfere with %r0 that is also being returned.
  // So we need to bufferize it out of place and make a new alloc.
  //  CHECK-DAG: %[[ALLOC:.*]] = memref.alloc({{.*}}) : memref<?xf32>
  //      CHECK: linalg.fill(%{{.*}}, %[[ALLOC]]
  %r1 = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
  //      CHECK: return %[[ALLOC]] : memref<?xf32>
  return %r1, %r0: tensor<?xf32>, tensor<?xf32>
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

  // CHECK-NEXT:   %[[A:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NEXT:   %[[B:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NEXT:   %[[C:.*]] = memref.alloc() : memref<f32>
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
