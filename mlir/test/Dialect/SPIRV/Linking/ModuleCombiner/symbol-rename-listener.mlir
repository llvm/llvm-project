// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

module {
spv.module @Module1 Logical GLSL450 {
  spv.GlobalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
  spv.func @bar() -> () "None" {
    spv.Return
  }
  spv.func @baz() -> () "None" {
    spv.Return
  }

  spv.SpecConstant @sc = -5 : i32
}

spv.module @Module2 Logical GLSL450 {
  spv.func @foo() -> () "None" {
    spv.Return
  }

  spv.GlobalVariable @bar bind(1, 0) : !spv.ptr<f32, Input>

  spv.func @baz() -> () "None" {
    spv.Return
  }

  spv.SpecConstant @sc = -5 : i32
}

spv.module @Module3 Logical GLSL450 {
  spv.func @foo() -> () "None" {
    spv.Return
  }

  spv.GlobalVariable @bar bind(1, 0) : !spv.ptr<f32, Input>

  spv.func @baz() -> () "None" {
    spv.Return
  }

  spv.SpecConstant @sc = -5 : i32
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
