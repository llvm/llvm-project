// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// Combine modules without the same symbols

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.SpecConstant @m1_sc
// CHECK-NEXT:     spv.GlobalVariable @m1_gv bind(1, 0)
// CHECK-NEXT:     spv.func @no_op
// CHECK-NEXT:       spv.Return
// CHECK-NEXT:     }
// CHECK-NEXT:     spv.EntryPoint "GLCompute" @no_op
// CHECK-NEXT:     spv.ExecutionMode @no_op "LocalSize", 32, 1, 1

// CHECK-NEXT:     spv.SpecConstant @m2_sc
// CHECK-NEXT:     spv.GlobalVariable @m2_gv bind(0, 1)
// CHECK-NEXT:     spv.func @variable_init_spec_constant
// CHECK-NEXT:       spv.mlir.referenceof @m2_sc
// CHECK-NEXT:       spv.Variable init
// CHECK-NEXT:       spv.Return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.SpecConstant @m1_sc = 42.42 : f32
  spv.GlobalVariable @m1_gv bind(1, 0): !spv.ptr<f32, Input>
  spv.func @no_op() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @no_op
  spv.ExecutionMode @no_op "LocalSize", 32, 1, 1
}

spv.module Logical GLSL450 {
  spv.SpecConstant @m2_sc = 42 : i32
  spv.GlobalVariable @m2_gv bind(0, 1): !spv.ptr<f32, Input>
  spv.func @variable_init_spec_constant() -> () "None" {
    %0 = spv.mlir.referenceof @m2_sc : i32
    %1 = spv.Variable init(%0) : !spv.ptr<i32, Function>
    spv.Return
  }
}
}

// -----

module {
spv.module Physical64 GLSL450 {
}

// expected-error @+1 {{input modules differ in addressing model, memory model, and/or VCE triple}}
spv.module Logical GLSL450 {
}
}

// -----

module {
spv.module Logical Simple {
}

// expected-error @+1 {{input modules differ in addressing model, memory model, and/or VCE triple}}
spv.module Logical GLSL450 {
}
}

// -----

module {
spv.module Logical GLSL450 {
}

// expected-error @+1 {{input modules differ in addressing model, memory model, and/or VCE triple}}
spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
}
}

