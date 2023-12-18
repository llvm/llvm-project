// RUN: mlir-opt %s --sparsification-and-bufferization | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  crdWidth = 32,
  posWidth = 32
}>

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = X(i,j) * 2"
}

//
// Pass in the buffers of the sparse tensor, marked non-writable.
// This forces a copy for the values and positions.
//
// CHECK-LABEL: func.func @foo(
// CHECK-SAME: %[[VAL:.*]]: memref<3xf64>,
// CHECK-SAME: %[[CRD:.*]]: memref<3xi32>,
// CHECK-SAME: %[[POS:.*]]: memref<11xi32>)
// CHECK:      %[[ALLOC1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xf64>
// CHECK:      memref.copy %[[VAL]], %[[ALLOC1]] : memref<3xf64> to memref<3xf64>
// CHECK:      %[[ALLOC2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<11xi32>
// CHECK:      memref.copy %[[POS]], %[[ALLOC2]] : memref<11xi32> to memref<11xi32>
// CHECK-NOT:  memref.copy
// CHECK:      return
//
func.func @foo(%arg0: tensor<3xf64>  {bufferization.writable = false},
               %arg1: tensor<3xi32>  {bufferization.writable = false},
               %arg2: tensor<11xi32> {bufferization.writable = false}) -> (index) {
    //
    // Pack the buffers into a sparse tensors.
    //
    %pack = sparse_tensor.assemble %arg0, %arg2, %arg1
      : tensor<3xf64>,
        tensor<11xi32>,
        tensor<3xi32> to tensor<10x10xf64, #CSR>

    //
    // Scale the sparse tensor "in-place" (this has no impact on the final
    // number of entries, but introduces reading the positions buffer
    // and writing into the value buffer).
    //
    %c = arith.constant 2.0 : f64
    %s = linalg.generic #trait_scale
      outs(%pack: tensor<10x10xf64, #CSR>) {
         ^bb(%x: f64):
          %1 = arith.mulf %x, %c : f64
          linalg.yield %1 : f64
    } -> tensor<10x10xf64, #CSR>

    //
    // Return number of entries in the scaled sparse tensor.
    //
    %nse = sparse_tensor.number_of_entries %s : tensor<10x10xf64, #CSR>
    return %nse : index
}

//
// Pass in the buffers of the sparse tensor, marked writable.
//
// CHECK-LABEL: func.func @bar(
// CHECK-SAME: %[[VAL:.*]]: memref<3xf64>,
// CHECK-SAME: %[[CRD:.*]]: memref<3xi32>,
// CHECK-SAME: %[[POS:.*]]: memref<11xi32>)
// CHECK-NOT:  memref.copy
// CHECK:      return
//
func.func @bar(%arg0: tensor<3xf64>  {bufferization.writable = true},
               %arg1: tensor<3xi32>  {bufferization.writable = true},
               %arg2: tensor<11xi32> {bufferization.writable = true}) -> (index) {
    //
    // Pack the buffers into a sparse tensors.
    //
    %pack = sparse_tensor.assemble %arg0, %arg2, %arg1
      : tensor<3xf64>,
        tensor<11xi32>,
        tensor<3xi32> to tensor<10x10xf64, #CSR>

    //
    // Scale the sparse tensor "in-place" (this has no impact on the final
    // number of entries, but introduces reading the positions buffer
    // and writing into the value buffer).
    //
    %c = arith.constant 2.0 : f64
    %s = linalg.generic #trait_scale
      outs(%pack: tensor<10x10xf64, #CSR>) {
         ^bb(%x: f64):
          %1 = arith.mulf %x, %c : f64
          linalg.yield %1 : f64
    } -> tensor<10x10xf64, #CSR>

    //
    // Return number of entries in the scaled sparse tensor.
    //
    %nse = sparse_tensor.number_of_entries %s : tensor<10x10xf64, #CSR>
    return %nse : index
}
