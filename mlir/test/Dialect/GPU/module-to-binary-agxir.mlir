// REQUIRES: host-supports-agx
// RUN: mlir-opt %s --gpu-module-to-binary="format=llvm" | FileCheck %s
// RUN: mlir-opt %s --gpu-module-to-binary="format=isa" | FileCheck %s -check-prefix=CHECK-ISA
// RUN: mlir-opt %s --gpu-module-to-binary="format=bin" | FileCheck %s -check-prefix=CHECK-BIN

module attributes {gpu.container_module} {
  // CHECK-LABEL: gpu.binary @kernel_module
  // CHECK: [#gpu.object<#agxir.target<chip = "agx-g13g">, offload = "{{.*}}">]
  // CHECK-ISA: [#gpu.object<#agxir.target<chip = "agx-g13g">, assembly = "{{.*}}device_store{{.*}}stop{{.*}}">]
  // CHECK-BIN: [#gpu.object<#agxir.target<chip = "agx-g13g">, bin = "{{.*}}ELF{{.*}}">]
  gpu.module @kernel_module [#agxir.target<chip = "agx-g13g">] {
    llvm.func agx_kernelcc @kernel(%out: !llvm.ptr<1>, %v: f32) attributes {agxir.kernel} {
      %id = agxir.thread.position.in.grid.x : i32
      %p = llvm.getelementptr %out[%id] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %p : f32, !llvm.ptr<1>
      llvm.return
    }
  }
}
