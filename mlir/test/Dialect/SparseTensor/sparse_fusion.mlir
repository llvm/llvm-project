// RUN: mlir-opt %s --linalg-fuse-elementwise-ops | FileCheck %s

#SV = #sparse_tensor.encoding<{ lvlTypes = ["compressed"] }>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>, // A
    affine_map<(i) -> (i)>  // B (out)
  ],
  iterator_types = ["parallel"],
  doc = "B(i) = OP A(i)"
}

// CHECK-LABEL: func @sparse_fusion
// CHECK:     linalg.generic
// CHECK:       arith.addf
// CHECK:     linalg.generic
// CHECK:       math.exp
// CHECK:       arith.maxf
// CHECK-NOT: linalg.generic
// CHECK:     return
func.func @sparse_fusion(%argA: tensor<100xf64, #SV>) -> tensor<100xf64> {
  %c1 = arith.constant 1.0 : f64
  %c100 = arith.constant 100.0 : f64

  //
  // Densifying op.
  // Should not be fused with subsequent dense ops.
  //
  %t0 = tensor.empty() : tensor<100xf64>
  %l0 = linalg.generic #trait
      ins(%argA: tensor<100xf64, #SV>) outs(%t0: tensor<100xf64>) {
    ^bb0(%in0: f64, %out0: f64):
      %b0 = arith.addf %in0, %c1 : f64
      linalg.yield %b0 : f64
  } -> tensor<100xf64>


  //
  // Two following dense ops.
  // Should be fused, but not with above.
  //
  %t1 = tensor.empty() : tensor<100xf64>
  %l1 = linalg.generic #trait
      ins(%l0: tensor<100xf64>) outs(%t1: tensor<100xf64>) {
    ^bb0(%in1: f64, %out1: f64):
      %b1 = math.exp %in1 : f64
      linalg.yield %b1 : f64
  } -> tensor<100xf64>
  %t2 = tensor.empty() : tensor<100xf64>
  %l2 = linalg.generic #trait
      ins(%l1: tensor<100xf64>) outs(%t2: tensor<100xf64>) {
    ^bb0(%in2: f64, %out2: f64):
      %b2 = arith.maxf %in2, %c100 : f64
      linalg.yield %b2 : f64
  } -> tensor<100xf64>

  return %l2 : tensor<100xf64>
}
