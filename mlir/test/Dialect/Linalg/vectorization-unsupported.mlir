// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics

// Masked vectorisation of `tensor.extract`:
//   * requires the `{ vectorize_nd_extract }` attribute,
//   * has not been implemented yet (hence the attribute is absent).
// TOOD: Implement masked vectorization for `tensor.extract`

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @extract_masked_vectorize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  // expected-error@+1 {{failed to vectorize op}}
  %2 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel"]
  } outs(%arg1 : tensor<?x?xf32>) {
  ^bb0(%arg3: f32):
    %7 = tensor.extract %arg0[%c0, %c1] : tensor<?x?xf32>
    linalg.yield %7 : f32
  } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}


transform.sequence failures(propagate) {
 ^bb1(%arg1: !pdl.operation):
   %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
   transform.structured.masked_vectorize %0 vector_sizes [3, 3]
 }
