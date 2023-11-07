// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.module
//===----------------------------------------------------------------------===//

// CHECK: module
spirv.module Logical GLSL450 {}

// CHECK: module @foo
spirv.module @foo Logical GLSL450 {}

// CHECK: module
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]> {}

// CHECK: module
spirv.module Logical GLSL450 {
	// CHECK-LABEL: llvm.func @empty()
  spirv.func @empty() -> () "None" {
		// CHECK: llvm.return
    spirv.Return
  }
}
