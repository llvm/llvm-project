// RUN: mlir-opt %s --gpu-to-llvm="use-bare-pointers-for-kernels=1" -split-input-file | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels [#nvvm.target]  {
    llvm.func @kernel_1(%arg0: f32, %arg1: !llvm.ptr<1>) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.mlir.constant(0 : index) : i64
      %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %5 = llvm.mlir.constant(10 : index) : i64
      %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.mlir.constant(1 : index) : i64
      %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
      llvm.return
    }
  }
  func.func @foo() {
    // CHECK: [[MEMREF:%.*]] = gpu.alloc () : memref<10xf32, 1>
    // CHECK: [[DESCRIPTOR:%.*]] = builtin.unrealized_conversion_cast [[MEMREF]] : memref<10xf32, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[PTR:%.*]] = llvm.extractvalue [[DESCRIPTOR]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: gpu.launch_func  @kernels::@kernel_1 blocks in ({{.*}}) threads in ({{.*}}) : i64
    // CHECK: args(%{{.*}} : f32, [[PTR]] : !llvm.ptr<1>)
    %0 = arith.constant 0. : f32
    %1 = gpu.alloc () : memref<10xf32, 1>
    %c8 = arith.constant 8 : index
    gpu.launch_func  @kernels::@kernel_1 blocks in (%c8, %c8, %c8) threads in (%c8, %c8, %c8) args(%0 : f32, %1 : memref<10xf32, 1>)
    return
  }
}
