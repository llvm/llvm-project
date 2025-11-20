// RUN: mlir-opt %s --gpu-to-llvm="use-bare-pointers-for-kernels=1 intersperse-sizes-for-kernels=1" -split-input-file | FileCheck %s

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
  llvm.func @malloc(i64) -> !llvm.ptr
  gpu.binary @kernels  [#gpu.object<#spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>, "">]
  func.func @main() attributes {llvm.emit_c_interface} {
    // CHECK: [[RANK1UMD:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %rank1UndefMemrefDescriptor = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: [[RANK2UMD:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %rank2UndefMemrefDescriptor = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %c1 = arith.constant 1 : index
    // CHECK: [[PTR1:%.*]] = llvm.extractvalue [[RANK1UMD]][1]
    // CHECK: [[PTR2:%.*]] = llvm.extractvalue [[RANK2UMD]][1]
    // CHECK: [[PTR3:%.*]] = llvm.extractvalue [[RANK2UMD]][1]
    // CHECK: [[SIZE1:%.*]] = llvm.mlir.constant(32 : index) : i64
    // CHECK: [[SIZE2:%.*]] = llvm.mlir.constant(256 : index) : i64
    // CHECK: [[SIZE3:%.*]] = llvm.mlir.constant(48 : index) : i64
    %6 = builtin.unrealized_conversion_cast %rank1UndefMemrefDescriptor : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<8xf32>
    %10 = builtin.unrealized_conversion_cast %rank2UndefMemrefDescriptor : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<8x8xi32>
    %14 = builtin.unrealized_conversion_cast %rank2UndefMemrefDescriptor : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x12xi8>
    // CHECK: gpu.launch_func  @kernels::@kernel_add blocks in ({{.*}}) threads in ({{.*}}) : i64 args([[PTR1]] : !llvm.ptr, [[SIZE1]] : i64, [[PTR2]] : !llvm.ptr, [[SIZE2]] : i64, [[PTR3]] : !llvm.ptr, [[SIZE3]] : i64)
    gpu.launch_func  @kernels::@kernel_add blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%6 : memref<8xf32>, %10 : memref<8x8xi32>, %14 : memref<4x12xi8>)
    return
  }
}
