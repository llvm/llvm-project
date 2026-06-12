// RUN: mlir-opt %s \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-gpu-to-llvm-spv=use-64bit-index=true \
// RUN:   --finalize-memref-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN:   --gpu-module-to-binary | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: gpu.binary @kernels
  // CHECK: [#gpu.object<#air.target, "MTLB{{.*}}">]
  gpu.module @kernels [#air.target] {
    gpu.func @saxpy(%a: f32, %x: memref<?xf32>, %y: memref<?xf32>, %n: index) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %i0 = arith.muli %bid, %bdim : index
      %i = arith.addi %i0, %tid : index
      %inb = arith.cmpi slt, %i, %n : index
      scf.if %inb {
        %xv = memref.load %x[%i] : memref<?xf32>
        %yv = memref.load %y[%i] : memref<?xf32>
        %ax = arith.mulf %a, %xv : f32
        %r  = arith.addf %ax, %yv : f32
        memref.store %r, %y[%i] : memref<?xf32>
      }
      gpu.return
    }
  }
}
