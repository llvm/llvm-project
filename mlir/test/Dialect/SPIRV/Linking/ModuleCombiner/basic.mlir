// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// Combine modules without the same symbols

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @m1_sc
// CHECK-NEXT:     spirv.GlobalVariable @m1_gv bind(1, 0)
// CHECK-NEXT:     spirv.func @no_op
// CHECK-NEXT:       spirv.Return
// CHECK-NEXT:     }
// CHECK-NEXT:     spirv.EntryPoint "GLCompute" @no_op
// CHECK-NEXT:     spirv.ExecutionMode @no_op "LocalSize", 32, 1, 1

// CHECK-NEXT:     spirv.SpecConstant @m2_sc
// CHECK-NEXT:     spirv.GlobalVariable @m2_gv bind(0, 1)
// CHECK-NEXT:     spirv.func @variable_init_spec_constant
// CHECK-NEXT:       spirv.mlir.referenceof @m2_sc
// CHECK-NEXT:       spirv.Variable init
// CHECK-NEXT:       spirv.Return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @m1_sc = 42.42 : f32
  spirv.GlobalVariable @m1_gv bind(1, 0): !spirv.ptr<f32, Input>
  spirv.func @no_op() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @no_op
  spirv.ExecutionMode @no_op "LocalSize", 32, 1, 1
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @m2_sc = 42 : i32
  spirv.GlobalVariable @m2_gv bind(0, 1): !spirv.ptr<f32, Input>
  spirv.func @variable_init_spec_constant() -> () "None" {
    %0 = spirv.mlir.referenceof @m2_sc : i32
    %1 = spirv.Variable init(%0) : !spirv.ptr<i32, Function>
    spirv.Return
  }
}
}

// -----

module {
spirv.module Physical64 GLSL450 {
}

// expected-error @+1 {{input modules differ in addressing model, memory model, and/or VCE triple}}
spirv.module Logical GLSL450 {
}
}

// -----

module {
spirv.module Logical Simple {
}

// expected-error @+1 {{input modules differ in addressing model, memory model, and/or VCE triple}}
spirv.module Logical GLSL450 {
}
}

// -----

module {
spirv.module Logical GLSL450 {
}

// expected-error @+1 {{input modules differ in addressing model, memory model, and/or VCE triple}}
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPIRV_KHR_storage_buffer_storage_class]> {
}
}

