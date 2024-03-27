// RUN: mlir-opt %s --canonicalize --pre-sparsification-rewrite | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#sparse = #sparse_tensor.encoding<{
  map = (d0, d1, d2) ->
          (d0 : compressed(nonunique),
	   d1 : singleton(nonunique, soa),
	   d2 : singleton(soa)),
  posWidth = 64,
  crdWidth = 64
}>


module {
  //
  // This IR should not end up in an infinite loop trying to fold
  // the linalg producer into the tensor cast consumer (even though
  // static sizes can fold, the different encodings cannot). The
  // cast was sloppy to begin with (but it has been observed by
  // external sources) and can be easily repaired by the sparsifier.
  //
  // CHECK-LABEL: func @avoid_fold
  // CHECK:       arith.constant
  // CHECK:       tensor.empty()
  // CHECK:       linalg.generic
  // CHECK:       sparse_tensor.convert
  // CHECK:       return
  //
  func.func @avoid_fold(%0: tensor<10x20x30xf64, #sparse>) -> tensor<10x20x30xf64, #sparse> {
    %1 = tensor.empty() : tensor<10x20x30xf64>
    %2 = linalg.generic { indexing_maps = [#map, #map],
                          iterator_types = ["parallel", "parallel", "parallel"]
                        }
    ins (%0 : tensor<10x20x30xf64, #sparse>)
    outs(%1 : tensor<10x20x30xf64>) {
        ^bb0(%in: f64, %out: f64):
          %cst = arith.constant 0.000000e+00 : f64
          %4 = arith.cmpf ugt, %in, %cst : f64
          %5 = arith.select %4, %in, %cst : f64
          linalg.yield %5 : f64
    } -> tensor<10x20x30xf64>
    %cast = tensor.cast %2 : tensor<10x20x30xf64> to tensor<10x20x30xf64, #sparse>
    return %cast : tensor<10x20x30xf64, #sparse>
  }
}

