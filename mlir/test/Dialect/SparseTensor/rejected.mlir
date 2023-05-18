// RUN: mlir-opt %s -sparsification | FileCheck %s


// The file contains examples that will be rejected by sparse compiler
// (we expect the linalg.generic unchanged).
#SparseVector = #sparse_tensor.encoding<{lvlTypes = ["compressed"]}>

#trait = {
  indexing_maps = [ 
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],  
  iterator_types = ["reduction"]
}

// CHECK-LABEL:   func.func @sparse_reduction_subi(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK:           %[[VAL_2:.*]] = linalg.generic
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32):
// CHECK:             %[[VAL_5:.*]] = arith.subi %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:             linalg.yield %[[VAL_5]] : i32
// CHECK:           } -> tensor<i32>
// CHECK:           return %[[VAL_6:.*]] : tensor<i32>
func.func @sparse_reduction_subi(%argx: tensor<i32>,
                             %arga: tensor<?xi32, #SparseVector>)
 -> tensor<i32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi32, #SparseVector>)
      outs(%argx: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        // NOTE: `subi %a, %x` is the reason why the program is rejected by the sparse compiler.
        // It is because we do not allow `-outTensor` in reduction loops as it creates cyclic
        // dependences.
        %t = arith.subi %a, %x: i32 
        linalg.yield %t : i32 
  } -> tensor<i32>
  return %0 : tensor<i32>
}
