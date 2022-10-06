// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

module {
spirv.module @Module1 Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
  spirv.func @bar() -> () "None" {
    spirv.Return
  }
  spirv.func @baz() -> () "None" {
    spirv.Return
  }

  spirv.SpecConstant @sc = -5 : i32
}

spirv.module @Module2 Logical GLSL450 {
  spirv.func @foo() -> () "None" {
    spirv.Return
  }

  spirv.GlobalVariable @bar bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @baz() -> () "None" {
    spirv.Return
  }

  spirv.SpecConstant @sc = -5 : i32
}

spirv.module @Module3 Logical GLSL450 {
  spirv.func @foo() -> () "None" {
    spirv.Return
  }

  spirv.GlobalVariable @bar bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @baz() -> () "None" {
    spirv.Return
  }

  spirv.SpecConstant @sc = -5 : i32
}
}

// CHECK: [Module1] foo -> foo_1
// CHECK: [Module1] sc -> sc_2

// CHECK: [Module2] bar -> bar_3
// CHECK: [Module2] baz -> baz_4
// CHECK: [Module2] sc -> sc_5

// CHECK: [Module3] foo -> foo_6
// CHECK: [Module3] bar -> bar_7
// CHECK: [Module3] baz -> baz_8
