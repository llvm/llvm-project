// REQUIRES: host-supports-amdgpu
// RUN: mlir-opt %s --gpu-module-to-binary="format=llvm" | FileCheck %s
// RUN: mlir-opt %s --gpu-module-to-binary="format=isa" | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.binary @kernel_module1
  // CHECK:[#gpu.object<#rocdl.target<chip = "gfx90a">, "{{.*}}">]
  gpu.module @kernel_module1 [#rocdl.target<chip = "gfx90a">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr<f32>,
        %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }

  // CHECK-LABEL:gpu.binary @kernel_module2
  // CHECK:[#gpu.object<#rocdl.target<flags = {fast}>, "{{.*}}">, #gpu.object<#rocdl.target, "{{.*}}">]
  gpu.module @kernel_module2 [#rocdl.target<flags = {fast}>, #rocdl.target] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr<f32>,
        %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
