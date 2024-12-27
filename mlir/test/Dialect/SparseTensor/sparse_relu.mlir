// RUN: mlir-opt %s --sparsification-and-bufferization | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#sparse = #sparse_tensor.encoding<{
    map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed)
}>

//
// Make sure a simple ReLU passes the sparsifier
//
// CHECK-LABEL: func.func @relu
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             arith.cmpf ugt
// CHECK:             arith.select
//
func.func @relu(%arg0: tensor<10x20x30xf64, #sparse>) -> tensor<10x20x30xf64, #sparse> {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = tensor.empty() : tensor<10x20x30xf64>
  %1 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<10x20x30xf64, #sparse>)
      outs(%0 : tensor<10x20x30xf64>) {
  ^bb0(%in: f64, %out: f64):
      %2 = arith.cmpf ugt, %in, %cst : f64
      %3 = arith.select %2, %in, %cst : f64
      linalg.yield %3 : f64
  } -> tensor<10x20x30xf64>
  %cast = tensor.cast %1 : tensor<10x20x30xf64> to tensor<10x20x30xf64, #sparse>
  return %cast : tensor<10x20x30xf64, #sparse>
}
