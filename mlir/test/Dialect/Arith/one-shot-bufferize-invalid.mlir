// RUN: mlir-opt %s -one-shot-bufferize -split-input-file -verify-diagnostics

// The two arms' buffers have layouts the default cast cannot cast to the type
// they are reconciled to: the select is not bufferized.

func.func @select_cast_refused(
    %c: i1, %a: memref<4xf32, affine_map<(d0) -> (d0 floordiv 2)>>,
    %b: memref<4xf32>) -> f32 {
  %ta = bufferization.to_tensor %a restrict
    : memref<4xf32, affine_map<(d0) -> (d0 floordiv 2)>> to tensor<4xf32>
  %tb = bufferization.to_tensor %b restrict : memref<4xf32> to tensor<4xf32>
  // expected-error @+2 {{cannot cast a buffer of type 'memref<4xf32, affine_map<(d0) -> (d0 floordiv 2)>>' to 'memref<4xf32, strided<[?], offset: ?>>'}}
  // expected-error @+1 {{failed to bufferize op}}
  %s = arith.select %c, %ta, %tb : tensor<4xf32>
  %c0 = arith.constant 0 : index
  %e = tensor.extract %s[%c0] : tensor<4xf32>
  return %e : f32
}
