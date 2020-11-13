// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.specConstant @m1_sc
// CHECK-NEXT:     spv.specConstant @m2_sc
// CHECK-NEXT:     spv.func @variable_init_spec_constant
// CHECK-NEXT:       spv._reference_of @m2_sc
// CHECK-NEXT:       spv.Variable init
// CHECK-NEXT:       spv.Return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.specConstant @m1_sc = 42.42 : f32
}

spv.module Logical GLSL450 {
  spv.specConstant @m2_sc = 42 : i32
  spv.func @variable_init_spec_constant() -> () "None" {
    %0 = spv._reference_of @m2_sc : i32
    %1 = spv.Variable init(%0) : !spv.ptr<i32, Function>
    spv.Return
  }
}
}

// -----

module {
spv.module Physical64 GLSL450 {
}

// expected-error @+1 {{input modules differ in addressing model and/or memory model}}
spv.module Logical GLSL450 {
}
}

// -----

module {
spv.module Logical Simple {
}

// expected-error @+1 {{input modules differ in addressing model and/or memory model}}
spv.module Logical GLSL450 {
}
}
