// RUN: mlir-opt %s -linalg-fold-unit-extent-dims -split-input-file | FileCheck %s

// Test that the init operand is preserved in outs() when it
// participates in an extract_slice -> generic -> insert_slice
// chain. Moving it to ins() would prevent bufferization from
// emitting an in-place update of the enclosing tensor.

// CHECK-LABEL: func @test1
// CHECK:         %[[IN:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0]
// CHECK:         %[[OUT:.*]] = tensor.extract_slice %{{.*}}[0]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[IN]] : tensor<64xf16>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<64xf16>)
// CHECK:         tensor.insert_slice

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @test1(%input: tensor<4x128xf16>, %acc: tensor<128xf16>, %iv: index) -> tensor<128xf16> {
  %in_slice = tensor.extract_slice %input[%iv, 0] [1, 64] [1, 1] : tensor<4x128xf16> to tensor<1x64xf16>
  %out_slice = tensor.extract_slice %acc[0] [64] [1] : tensor<128xf16> to tensor<64xf16>
  %generic = linalg.generic {
      indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]}
      ins(%in_slice : tensor<1x64xf16>) outs(%out_slice : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %max = arith.maxnumf %in, %out : f16
      linalg.yield %max : f16
  } -> tensor<64xf16>
  %result = tensor.insert_slice %generic into %acc[0] [64] [1] : tensor<64xf16> into tensor<128xf16>
  return %result : tensor<128xf16>
}

// -----

// Test that the unit-dim fold still moves the init operand to ins()
// and replaces outs() with tensor.empty() when there is no surrounding
// extract_slice -> insert_slice chain.

// CHECK-LABEL: func @test2
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<64xf16>)

#map2 = affine_map<(d0) -> (d0)>

func.func @test2(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %cst = arith.constant 0xFC00 : f16
  %init = linalg.fill ins(%cst : f16) outs(%arg1 : tensor<64xf16>) -> tensor<64xf16>
  %result = linalg.generic {
      indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %add = arith.addf %in, %out : f16
      linalg.yield %add : f16
  } -> tensor<64xf16>
  return %result : tensor<64xf16>
}
