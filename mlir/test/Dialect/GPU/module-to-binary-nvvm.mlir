// REQUIRES: host-supports-nvptx
// RUN: mlir-opt %s --gpu-module-to-binary="format=llvm" | FileCheck %s
// RUN: mlir-opt %s --gpu-module-to-binary="format=isa" | FileCheck %s -check-prefix=CHECK-ISA

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.binary @kernel_module1
  // CHECK:[#gpu.object<#nvvm.target<chip = "sm_70">, offload = "{{.*}}">]
  gpu.module @kernel_module1 [#nvvm.target<chip = "sm_70">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr<f32>,
        %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }

  // CHECK-LABEL:gpu.binary @kernel_module2
  // CHECK-ISA:[#gpu.object<#nvvm.target<flags = {fast}>, properties = {O = 2 : i32}, assembly = "{{.*}}">, #gpu.object<#nvvm.target, properties = {O = 2 : i32}, assembly = "{{.*}}">]
  gpu.module @kernel_module2 [#nvvm.target<flags = {fast}>, #nvvm.target] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr<f32>,
        %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
