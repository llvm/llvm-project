// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

// CHECK: module
spv.module Logical GLSL450 {}

// CHECK: module @foo
spv.module @foo Logical GLSL450 {}

// CHECK: module
spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]> {}

// CHECK: module
spv.module Logical GLSL450 {
	// CHECK-LABEL: llvm.func @empty()
  spv.func @empty() -> () "None" {
		// CHECK: llvm.return
    spv.Return
  }
}
