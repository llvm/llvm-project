// RUN: mlir-opt %s -test-side-effects -verify-diagnostics

func.func @sincos_effects(%x: f32, %cosval: !spirv.ptr<f32, Function>) -> f32 {
  // expected-remark@+1 {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  %sin = spirv.CL.sincos %x, %cosval : f32, !spirv.ptr<f32, Function> -> f32
  return %sin : f32
}
