// RUN: mlir-opt -infer-static-shapes -split-input-file %s | FileCheck %s

// CHECK-LABEL:  func.func @pad_reification
func.func @pad_reification(%cst : f32, %idx : index, %t: tensor<64x?x64xf32>)
    -> tensor<1x?x64xf32> {
  %pad_amt = affine.apply affine_map<(d0) -> (-d0 + 256)>(%idx)
  %es = tensor.extract_slice %t[0, 0, 0] [1, %idx, 64] [1, 1, 1] 
    : tensor<64x?x64xf32> to tensor<1x?x64xf32>

//       CHECK: tensor.pad
//       CHECK:   : tensor<1x?x64xf32> to tensor<1x256x64xf32>
  %padded = tensor.pad %es low[0, 0, 0] high[0, %pad_amt, 0] {
    ^bb0(%a: index, %b: index, %c: index):
    tensor.yield %cst : f32
  } : tensor<1x?x64xf32> to tensor<1x?x64xf32>

  return %padded : tensor<1x?x64xf32>
}
