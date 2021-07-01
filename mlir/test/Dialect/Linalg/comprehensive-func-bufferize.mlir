// RUN: mlir-opt %s -linalg-comprehensive-func-bufferize -split-input-file | FileCheck %s

// CHECK-DAG: #[[$map_2d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @fill_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: tensor<?xf32> {linalg.inplaceable = true})
func @fill_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}) -> tensor<?xf32> {
  //     CHECK: %[[I:.*]] = memref.buffer_cast %[[A]] : memref<?xf32, #[[$map_2d_dyn]]>

  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

  /// Inplaceable, no alloc
  // CHECK-NOT: alloc
  //     CHECK: linalg.fill(%[[F0]], %[[I]]) : f32, memref<?xf32, #[[$map_2d_dyn]]>
  %r = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //     CHECK:  %[[R:.*]] = memref.tensor_load %[[I]] : memref<?xf32, #[[$map_2d_dyn]]>
  //     CHECK:  return %[[R]] : tensor<?xf32>
  return %r: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_2d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

/// No linalg.inplaceable flag, must allocate.
// CHECK-LABEL: func @not_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: tensor<?xf32>)
func @not_inplace(%A : tensor<?xf32>) -> tensor<?xf32> {
  //     CHECK: %[[I:.*]] = memref.buffer_cast %[[A]] : memref<?xf32, #[[$map_2d_dyn]]>

  //     CHECK: %[[D0:.*]] = memref.dim %[[I]], {{.*}} : memref<?xf32, #[[$map_2d_dyn]]>
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[D0]]) : memref<?xf32>
  //     CHECK: %[[I2:.*]] = memref.cast %[[ALLOC]] : memref<?xf32> to memref<?xf32, #map>

  //     CHECK: %[[F0:.*]] = constant 0.000000e+00 : f32
  %f0 = constant 0.0 : f32

  //     CHECK: linalg.fill(%[[F0]], %[[I2]]) : f32, memref<?xf32, #[[$map_2d_dyn]]>
  %r = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //     CHECK:  dealloc %[[ALLOC]] : memref<?xf32>
  //     CHECK:  %[[R:.*]] = memref.tensor_load %[[I2]] : memref<?xf32, #[[$map_2d_dyn]]>
  //     CHECK:  return %[[R]] : tensor<?xf32>
  return %r: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: tensor<?x?xf32>
func @not_inplace(%A : tensor<?x?xf32> {linalg.inplaceable = true}) -> tensor<?x?xf32> {
  %f0 = constant 0.0 : f32

  //       CHECK: %[[BUFFER_CAST:.*]] = memref.buffer_cast %[[A]] : memref<?x?xf32

  /// Cross-op multiple uses of %A, the first op which has interfering reads must alloc.
  //       CHECK: %[[ALLOC:.*]] = memref.alloc
  //       CHECK: %[[CAST:.*]] = memref.cast %[[ALLOC]]
  //       CHECK: linalg.fill({{.*}}, %[[CAST]]
  %f = linalg.fill(%f0, %A) : f32, tensor<?x?xf32> -> tensor<?x?xf32>

  /// The second op has no interfering reads and can reuse.
  //   CHECK-NOT: alloc
  //       CHECK: linalg.matmul{{.*}}outs(%[[BUFFER_CAST]]
  %r = linalg.matmul  ins(%f, %f: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>
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
  return %r: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @vec_not_inplace
func @vec_not_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %vec : vector<4xf32>)
    -> (tensor<?xf32>, tensor<?xf32>)
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  //       CHECK: %[[BUFFER_CAST:.*]] = memref.buffer_cast {{.*}} : memref<?xf32, #[[$map_2d_dyn]]>

  /// Cross-op multiple uses of %A, the first vector.transfer which has interfering reads must alloc.
  //      CHECK: %[[ALLOC:.*]] = memref.alloc
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[ALLOC]]
  %r0 = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>

  /// The second vector.transfer has no interfering reads and can reuse the buffer.
  //  CHECK-NOT: alloc
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[BUFFER_CAST]]
  %r1 = vector.transfer_write %vec, %A[%c1] : vector<4xf32>, tensor<?xf32>
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun
func @insert_slice_fun(%A0 : tensor<?xf32>, %A1 : tensor<?xf32> {linalg.inplaceable = true},
                           %t0 : tensor<4xf32>, %t1 : tensor<4xf32> {linalg.inplaceable = true})
  ->  (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
{
  //      CHECK: %[[BUFFER_CAST_A0:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_A1:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_t0:.*]] = memref.buffer_cast {{.*}} : memref<4xf32
  //      CHECK: %[[BUFFER_CAST_t1:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC_A0:.*]] = memref.alloc
  //      CHECK: linalg.copy(%[[BUFFER_CAST_A0]]
  //      CHECK: %[[SV_A0:.*]] = memref.subview %[[REALLOC_A0]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_t0]], %[[SV_A0]])
  %r0 = tensor.insert_slice %t0 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // Alloc and copy the whole result tensor. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC_A0_2:.*]] = memref.alloc
  //      CHECK: linalg.copy(%[[BUFFER_CAST_A0]]
  //      CHECK: %[[SV_A0_2:.*]] = memref.subview %[[REALLOC_A0_2]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_t1]], %[[SV_A0_2]])
  %r1 = tensor.insert_slice %t1 into %A0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Still alloc the large tensor because %A1 is read after. Copy the tensor.extract_slice.
  //      CHECK: %[[REALLOC_A1:.*]] = memref.alloc
  //      CHECK: linalg.copy(%[[BUFFER_CAST_A1]]
  //      CHECK: %[[SV_A1:.*]] = memref.subview %[[REALLOC_A1]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_t0]], %[[SV_A1]])
  %r2 = tensor.insert_slice %t0 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //  Do not realloc the large tensor. Copy the tensor.extract_slice.
  //  CHECK-NOT: alloc
  //      CHECK: %[[SV_A1_2:.*]] = memref.subview %[[BUFFER_CAST_A1]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_t1]], %[[SV_A1_2]])
  %r3 = tensor.insert_slice %t1 into %A1[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r0, %r1, %r2, %r3: tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun
func @insert_slice_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  %f0 = constant 0.0 : f32

  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]]
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]])
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  /// Overwrite BUFFER_CAST_A inplace.
  //      CHECK: linalg.fill({{.*}}, %[[BUFFER_CAST_A]]
  %r1 = linalg.fill(%f0, %r0) : f32, tensor<?xf32> -> tensor<?xf32>
  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun
func @insert_slice_fun(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  %f0 = constant 0.0 : f32

  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //      CHECK: linalg.fill({{.*}}, %[[BUFFER_CAST_A]]
  %r0 = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //  CHECK-NOT: alloc
  //      CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]]
  /// Overwrite BUFFER_CAST_A inplace by copying into the subview.
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]])
  %r1 = tensor.insert_slice %t into %r0[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun_not_inplace
func @insert_slice_fun_not_inplace(%A : tensor<?xf32>, %t : tensor<4xf32>)
  -> tensor<?xf32>
{
  //      CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //      CHECK: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32

  //      CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  //      CHECK: linalg.copy(%[[BUFFER_CAST_A]], %[[ALLOC]]) : memref<?xf32{{.*}}, memref<?xf32>
  //      CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]][0] [4] [1] : memref<?xf32> to memref<4xf32>
  //      CHECK: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]]) : memref<4xf32, #map>, memref<4xf32>
  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>
  return %r0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun_not_inplace
func @insert_slice_fun_not_inplace(%A : tensor<?xf32> {linalg.inplaceable = true}, %t : tensor<4xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %f0 = constant 0.0 : f32

  //  CHECK-DAG: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32{{.*}}
  //  CHECK-DAG: %[[BUFFER_CAST_B:.*]] = memref.buffer_cast {{.*}} : memref<4xf32{{.*}}

  // tensor.insert_slice is bufferized first, %A is inplaceable so we can make this inplace
  //  CHECK-DAG: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]][0] [4] [1] : memref<?xf32, {{.*}}> to memref<4xf32, {{.*}}>
  //  CHECK-DAG: linalg.copy(%[[BUFFER_CAST_B]], %[[SV]]) : memref<4xf32, {{.*}}>, memref<4xf32, {{.*}}>
  %r0 = tensor.insert_slice %t into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // fill would interfere with %r0 that is also being returned.
  // So we need to bufferize it out of place and make a new alloc.
  //  CHECK-DAG: %[[ALLOC:.*]] = memref.alloc({{.*}}) : memref<?xf32>
  //  CHECK-DAG: %[[ALLOC_CAST_DYNAMIC:.*]] = memref.cast %[[ALLOC]] : memref<?xf32> to memref<?xf32, {{.*}}
  //      CHECK: linalg.fill(%{{.*}}, %[[ALLOC_CAST_DYNAMIC]]
  //      CHECK: memref.dealloc %[[ALLOC]] : memref<?xf32>
  %r1 = linalg.fill(%f0, %A) : f32, tensor<?xf32> -> tensor<?xf32>

  //  CHECK-DAG: %[[RET_A:.*]] = memref.tensor_load %[[BUFFER_CAST_A]] : memref<?xf32, {{.*}}
  //  CHECK-DAG: %[[RET_B:.*]] = memref.tensor_load %[[ALLOC_CAST_DYNAMIC]] : memref<?xf32, {{.*}}
  //      CHECK: return %[[RET_B]], %[[RET_A]]
  return %r1, %r0: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_fun
func @extract_slice_fun(%A : tensor<?xf32> {linalg.inplaceable = true})
  ->  tensor<4xf32>
{
  // This bufferizes to a pattern that the cross-function boundary pass needs to
  // convert into a new memref argument at all call site; this may be either:
  //   - an externally created aliasing subview (if we want to allow aliasing
  //     function arguments).
  //   - a new alloc + copy (more expensive but does not create new function
  //     argument aliasing).
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  //     CHECK: %[[BUFFER_CAST_A:.*]] = memref.buffer_cast {{.*}} : memref<?xf32
  //     CHECK: %[[SV:.*]] = memref.subview %[[BUFFER_CAST_A]][0] [4] [1]
  //     CHECK: %[[RES:.*]] = memref.tensor_load %[[SV]]
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //     CHECK: return %[[RES]]
  return %r0: tensor<4xf32>
}

//===----------------------------------------------------------------------===//
// Simple loop cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @scf_for_yield_only
func @scf_for_yield_only(%A : tensor<?xf32>,
                         %B : tensor<?xf32> {linalg.inplaceable = true},
                         %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   %[[BUFFER_CAST_A:.*]] = memref.buffer_cast
  //     CHECK:   %[[BUFFER_CAST_B:.*]] = memref.buffer_cast
  //     CHECK:   linalg.copy(%[[BUFFER_CAST_A]], %[[ALLOC_FOR_A]])

  // The first scf.for remains but just turns into dead code.
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  // The second scf.for remains but just turns into dead code.
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  // Cross function call alloc/dealloc pattern must be hoist out.
  //     CHECK:   memref.dealloc %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:   %[[rA:.*]] = memref.tensor_load %[[ALLOC_FOR_A]]
  // Returning tensor_load of the buffer cast makes the %r1 loop dead.
  //     CHECK:   %[[rB:.*]] = memref.tensor_load %[[BUFFER_CAST_B:.*]]
  //     CHECK:   return %[[rA]], %[[rB]] : tensor<?xf32>, tensor<?xf32>
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
func @scf_for_with_tensor.insert_slice(
   %A : tensor<?xf32>,
              %B : tensor<?xf32> {linalg.inplaceable = true},
              %C : tensor<4xf32>,
              %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   %[[BUFFER_CAST_A:.*]] = memref.buffer_cast
  //     CHECK:   %[[BUFFER_CAST_B:.*]] = memref.buffer_cast
  //     CHECK:   %[[BUFFER_CAST_C:.*]] = memref.buffer_cast
  //     CHECK:   linalg.copy(%[[BUFFER_CAST_A]], %[[ALLOC_FOR_A]])

  //     CHECK:   scf.for {{.*}} iter_args(%[[bbA:.*]] = %{{.*}}, %[[bbB:.*]] = %{{.*}})
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    //     CHECK: %[[svA:.*]] = memref.subview %[[ALLOC_FOR_A]][0] [4] [1]
    // %ttA bufferizes to direct copy of %BUFFER_CAST_C into %svA
    //     CHECK: linalg.copy(%[[BUFFER_CAST_C]], %[[svA]])
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // %ttB bufferizes to direct copy of %BUFFER_CAST_C into %BUFFER_CAST_B
    //     CHECK:   %[[svB:.*]] = memref.subview %[[BUFFER_CAST_B]][0] [4] [1]
    //     CHECK:   linalg.copy(%[[BUFFER_CAST_C]], %[[svB]])
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // Yielding bbA and bbB will canonicalize away into oblivion.
    //     CHECK:   scf.yield %[[bbA]], %[[bbB]] : tensor<?xf32>, tensor<?xf32>
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  //     CHECK:  memref.dealloc %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:  %[[rA:.*]] = memref.tensor_load %[[ALLOC_FOR_A]] : memref<?xf32>
  //     CHECK:  %[[rB:.*]] = memref.tensor_load %[[BUFFER_CAST_B]] : memref<?xf32, #map>
  //     CHECK:  return %[[rA]], %[[rB]] : tensor<?xf32>, tensor<?xf32>
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}
