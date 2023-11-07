// RUN: mlir-opt %s -convert-gpu-to-nvvm -cse -canonicalize | FileCheck %s

gpu.module @modules {
  // CHECK: llvm.mlir.global internal @__shmem_dynamic_shared_memory_kernel_0() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x f32>
  
  // CHECK-LABEL: llvm.func @dynamic_shared_memory_kernel(
  // CHECK-SAME: %[[arg0:.+]]: i64)
  gpu.func @dynamic_shared_memory_kernel(%d : index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {    
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = gpu.dynamic.shared.memory [1, 0, 0] : memref<32x64xf32, #gpu.address_space<workgroup>>
    %1 = gpu.dynamic.shared.memory [1, 0, 0] : memref<32x32xf32, 3>
    %2 = gpu.dynamic.shared.memory [4, 234] : memref<32x32xf32, #gpu.address_space<workgroup>>
    %3 = gpu.dynamic.shared.memory [%c100, 4] : memref<32x32xf32, #gpu.address_space<workgroup>>
    %4 = gpu.dynamic.shared.memory [%c100, %d] : memref<128x64xf16, #gpu.address_space<workgroup>>
    %5 = gpu.dynamic.shared.memory [32, 0, 0] : memref<32x8xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    "test.use.shared.memory"(%1) : (memref<32x32xf32, 3>) -> ()
    "test.use.shared.memory"(%2) : (memref<32x32xf32, #gpu.address_space<workgroup>>) -> ()
    "test.use.shared.memory"(%3) : (memref<32x32xf32, #gpu.address_space<workgroup>>) -> ()
    "test.use.shared.memory"(%4) : (memref<128x64xf16, #gpu.address_space<workgroup>>) -> ()
    "test.use.shared.memory"(%5) : (memref<32x8xf32, #gpu.address_space<workgroup>>) -> ()

    // CHECK: %[[S2:.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK: %[[S3:.+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: %[[S4:.+]] = llvm.mlir.constant(64 : index) : i64
    // CHECK: %[[S5:.+]] = llvm.mlir.constant(32 : index) : i64
    // CHECK: %[[S6:.+]] = llvm.mlir.addressof @__shmem_dynamic_shared_memory_kernel_0 : !llvm.ptr<3>
    // CHECK: %[[S7:.+]] = llvm.getelementptr %[[S6]][1, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<32 x array<64 x f32>>
    // CHECK: %[[S8:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[S9:.+]] = llvm.insertvalue %[[S7]], %[[S8]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S10:.+]] = llvm.insertvalue %[[S7]], %[[S9]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S11:.+]] = llvm.insertvalue %[[S2]], %[[S10]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S12:.+]] = llvm.insertvalue %[[S5]], %[[S11]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S13:.+]] = llvm.insertvalue %[[S4]], %[[S12]][3, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S14:.+]] = llvm.insertvalue %[[S4]], %[[S13]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S15:.+]] = llvm.insertvalue %[[S3]], %[[S14]][4, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S16:.+]] = builtin.unrealized_conversion_cast %[[S15]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32, #gpu.address_space<workgroup>>

    // CHECK: %[[S17:.+]] = llvm.getelementptr %[[S6]][1, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<32 x array<32 x f32>>
    // CHECK: %[[S25:.+]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x32xf32, 3>

    // CHECK: %[[S26:.+]] = llvm.getelementptr %[[S6]][4, 234] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<32 x array<32 x f32>>
    // CHECK: %[[S34:.+]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x32xf32, #gpu.address_space<workgroup>>

    // CHECK: %[[S35:.+]] = llvm.getelementptr %[[S6]][100, 4] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<32 x array<32 x f32>>
    // CHECK: %[[S43:.+]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x32xf32, #gpu.address_space<workgroup>>

    // CHECK: %[[S44:.+]] = llvm.getelementptr %[[S6]][100, %[[arg0]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<128 x array<64 x f16>>
    // CHECK: %[[S52:.+]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<128x64xf16, #gpu.address_space<workgroup>>

    // CHECK: %[[S53:.+]] = llvm.getelementptr %[[S6]][32, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<32 x array<8 x f32>>
    // CHECK: %[[S61:.+]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x8xf32, #gpu.address_space<workgroup>>

    // CHECK: "test.use.shared.memory"(%[[S16]]) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    // CHECK: "test.use.shared.memory"(%[[S25]]) : (memref<32x32xf32, 3>) -> ()
    // CHECK: "test.use.shared.memory"(%[[S34]]) : (memref<32x32xf32, #gpu.address_space<workgroup>>) -> ()
    // CHECK: "test.use.shared.memory"(%[[S43]]) : (memref<32x32xf32, #gpu.address_space<workgroup>>) -> ()
    // CHECK: "test.use.shared.memory"(%[[S52]]) : (memref<128x64xf16, #gpu.address_space<workgroup>>) -> ()
    // CHECK: "test.use.shared.memory"(%[[S61]]) : (memref<32x8xf32, #gpu.address_space<workgroup>>) -> ()

    gpu.return
  }
}