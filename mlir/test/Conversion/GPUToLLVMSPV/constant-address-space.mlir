// RUN: mlir-opt %s -convert-gpu-to-llvm-spv | FileCheck %s

gpu.module @kernels {
  // CHECK-LABEL: llvm.func spir_kernelcc @constant_load
  // Constant address space maps to SPIRV/OpenCL address space 2 (UniformConstant)
  // CHECK-SAME: !llvm.ptr<2>
  gpu.func @constant_load(%arg0: memref<16xf32, #gpu.address_space<constant>>) kernel {
    %c0 = arith.constant 0 : index
    %v = memref.load %arg0[%c0] : memref<16xf32, #gpu.address_space<constant>>
    gpu.return
  }

  // CHECK-LABEL: llvm.func spir_funccc @all_address_spaces
  // Global -> 1, Workgroup -> 3, Private -> 0 (default), Constant -> 2
  // CHECK-SAME: !llvm.ptr<1>
  // CHECK-SAME: !llvm.ptr<3>
  // CHECK-SAME: !llvm.ptr,
  // CHECK-SAME: !llvm.ptr<2>
  gpu.func @all_address_spaces(
    %arg0: memref<f32, #gpu.address_space<global>>,
    %arg1: memref<f32, #gpu.address_space<workgroup>>,
    %arg2: memref<f32, #gpu.address_space<private>>,
    %arg3: memref<f32, #gpu.address_space<constant>>) {
    gpu.return
  }
}
