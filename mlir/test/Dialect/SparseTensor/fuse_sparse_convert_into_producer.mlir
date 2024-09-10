// RUN: mlir-opt %s --pre-sparsification-rewrite --sparse-reinterpret-map  | FileCheck %s --check-prefix=CHECK-FOLD
// RUN: mlir-opt %s --pre-sparsification-rewrite --sparse-reinterpret-map --sparsification | FileCheck %s

#trait = {
  indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#COO = #sparse_tensor.encoding<{map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, soa), d2 : singleton(soa))}>
#CCCD = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : dense) }>

// CHECK-LABEL:   func.func @fold_convert(
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.if
// CHECK-NEXT:               tensor.insert
// CHECK-NEXT:               scf.yield
// CHECK-NEXT:             else
// CHECK-NEXT:               scf.yield
// CHECK:                 scf.yield
// CHECK:               scf.yield
// CHECK:             scf.yield
// CHECK:           sparse_tensor.load

// CHECK-FOLD-LABEL:   func.func @fold_convert(
// CHECK-FOLD-NOT:     sparse_tensor.convert
func.func @fold_convert(%arg0: tensor<128x32x32x1xf32>, %arg1: tensor<128x32x32x1xf32>, %arg2: tensor<128x32x32x1xf32>) -> tensor<128x32x32x1xf32, #CCCD> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x32x32x1xf32>
  %1 = linalg.generic #trait
  ins(%arg0, %arg1, %arg2 : tensor<128x32x32x1xf32>, tensor<128x32x32x1xf32>, tensor<128x32x32x1xf32>)
  outs(%0 : tensor<128x32x32x1xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %3 = arith.subf %cst_0, %in_2 : f32
      %4 = arith.mulf %in, %3 : f32
      %5 = arith.mulf %4, %cst_1 : f32
      %6 = arith.addf %5, %in_3 : f32
      %7 = arith.subf %6, %cst_0 : f32
      %8 = arith.cmpf uge, %7, %cst : f32
      %9 = arith.uitofp %8 : i1 to f32
      linalg.yield %9 : f32
    } -> tensor<128x32x32x1xf32>
  %2 = sparse_tensor.convert %1 : tensor<128x32x32x1xf32> to tensor<128x32x32x1xf32, #CCCD>
  return %2 : tensor<128x32x32x1xf32, #CCCD>
}

#trait_bin = {
  indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

// CHECK-FOLD-LABEL:   func.func @fold_convert_multi_use(
// CHECK-FOLD:           tensor.empty() : tensor<128x32x32x1xf32>
// CHECK-FOLD:           linalg.generic
// CHECK-FOLD:           tensor.empty() : tensor<128x32x32x1xf32, #sparse>
// CHECK-FOLD:           linalg.generic
// CHECK-FOLD-NOT:       sparse_tensor.convert
func.func @fold_convert_multi_use(%arg0: tensor<128x32x32x1xf32>, %arg1: tensor<128x32x32x1xf32>,
                        %arg2: tensor<128x32x32x1xf32>, %arg3: tensor<128x32x32x1xf32>) -> (tensor<128x32x32x1xf32>, tensor<128x32x32x1xf32, #CCCD>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32

  %0 = tensor.empty() : tensor<128x32x32x1xf32>
  %1 = linalg.generic #trait_bin
  ins(%arg0, %arg1 : tensor<128x32x32x1xf32>, tensor<128x32x32x1xf32>)
  outs(%0 : tensor<128x32x32x1xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<128x32x32x1xf32>

  // A second kernel that uses %0 as the init operand.
  %3 = linalg.generic #trait_bin
  ins(%arg2, %arg3 : tensor<128x32x32x1xf32>, tensor<128x32x32x1xf32>)
  outs(%0 : tensor<128x32x32x1xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<128x32x32x1xf32>
  %4 = sparse_tensor.convert %3 : tensor<128x32x32x1xf32> to tensor<128x32x32x1xf32, #CCCD>

  return %1, %4 : tensor<128x32x32x1xf32>, tensor<128x32x32x1xf32, #CCCD>
}



// FIXME: The following kernel is not sparsifiable because `arith.select`
// operations is not handled by the sparse compiler at the moment.
//
// CHECK-FOLD-LABEL:   func.func @fold_cast(
// CHECK-FOLD-NOT:     sparse_tensor.convert
func.func @fold_cast(%0: tensor<10x20x30xf64, #COO>) -> tensor<10x20x30xf64, #COO> {
  %cst = arith.constant 0.000000e+00 : f64
  %1 = tensor.empty() : tensor<10x20x30xf64>
  %2 = linalg.generic { indexing_maps = [#map, #map],
                        iterator_types = ["parallel", "parallel", "parallel"]
                      }
  ins (%0 : tensor<10x20x30xf64, #COO>)
  outs(%1 : tensor<10x20x30xf64>) {
      ^bb0(%in: f64, %out: f64):
        %4 = arith.cmpf ugt, %in, %cst : f64
        %5 = arith.select %4, %in, %cst : f64
        linalg.yield %5 : f64
  } -> tensor<10x20x30xf64>
  %cast = tensor.cast %2 : tensor<10x20x30xf64> to tensor<10x20x30xf64, #COO>
  return %cast : tensor<10x20x30xf64, #COO>
}
