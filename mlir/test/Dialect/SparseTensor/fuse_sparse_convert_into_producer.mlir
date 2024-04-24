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

#sparse = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : dense) }>

// CHECK-LABEL:   func.func @test(
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
func.func @test(%arg0: tensor<128x32x32x1xf32>, %arg1: tensor<128x32x32x1xf32>, %arg2: tensor<128x32x32x1xf32>) -> tensor<128x32x32x1xf32, #sparse> {
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
  %2 = sparse_tensor.convert %1 : tensor<128x32x32x1xf32> to tensor<128x32x32x1xf32, #sparse>
  return %2 : tensor<128x32x32x1xf32, #sparse>
}
