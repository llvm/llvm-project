// RUN: mlir-opt %s -convert-gpu-to-nvvm -cse -canonicalize | FileCheck %s

gpu.module @modules {
  // CHECK: llvm.mlir.global internal @__dynamic_shmem__3() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.mlir.global internal @__dynamic_shmem__0() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<0 x i8>
  llvm.mlir.global internal @__dynamic_shmem__1() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<0 x i8>  
  llvm.mlir.global internal @__dynamic_shmem__2() {alignment = 16 : i64} : !llvm.array<0 x i8>  
  // CHECK-LABEL: llvm.func @dynamic_shared_memory_kernel(
  // CHECK-SAME: %[[arg0:.+]]: i64)
  gpu.func @dynamic_shared_memory_kernel(%d : index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {    
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %c16384 = arith.constant 16384 : index
    %shmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %shmem2 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>

    %0 = memref.view %shmem[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()

    %1 = memref.view %shmem[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%1) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    
// CHECK: %[[S0:.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK: %[[S1:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK: %[[S2:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[S3:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[S4:.+]] = llvm.mlir.addressof @__dynamic_shmem__3 : !llvm.ptr<3>
// CHECK: %[[S5:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[S6:.+]] = llvm.insertvalue %[[S4]], %[[S5]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S7:.+]] = llvm.getelementptr %[[S4]][8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: %[[S8:.+]] = llvm.insertvalue %[[S7]], %[[S6]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S9:.+]] = llvm.insertvalue %[[S3]], %[[S8]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S10:.+]] = llvm.insertvalue %[[S1]], %[[S9]][3, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S11:.+]] = llvm.insertvalue %[[S2]], %[[S10]][4, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S12:.+]] = llvm.insertvalue %[[S0]], %[[S11]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S13:.+]] = llvm.insertvalue %[[S1]], %[[S12]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S14:.+]] = builtin.unrealized_conversion_cast %[[S13]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32, #gpu.address_space<workgroup>>
// CHECK: "test.use.shared.memory"(%[[S14]]) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
// CHECK: %[[S15:.+]] = llvm.getelementptr %4[16384] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: %[[S16:.+]] = llvm.insertvalue %[[S15]], %[[S6]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S17:.+]] = llvm.insertvalue %[[S3]], %[[S16]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S18:.+]] = llvm.insertvalue %[[S1]], %[[S17]][3, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S19:.+]] = llvm.insertvalue %[[S2]], %[[S18]][4, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S20:.+]] = llvm.insertvalue %[[S0]], %[[S19]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S21:.+]] = llvm.insertvalue %[[S1]], %[[S20]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S22:.+]] = builtin.unrealized_conversion_cast %[[S21]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32, #gpu.address_space<workgroup>>
// CHECK: "test.use.shared.memory"(%[[S22]]) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
    gpu.return
  }

// CHECK-LABEL: llvm.func @gpu_device_function
  gpu.func @gpu_device_function()  {    
    %c8192 = arith.constant 8192 : index
    %shmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %0 = memref.view %shmem[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
// CHECK: %[[S0:.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK: %[[S1:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK: %[[S2:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[S3:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[S4:.+]] = llvm.mlir.addressof @__dynamic_shmem__3 : !llvm.ptr<3>
// CHECK: %[[S5:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[S6:.+]] = llvm.insertvalue %[[S4]], %[[S5]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S7:.+]] = llvm.getelementptr %[[S4]][8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: %[[S8:.+]] = llvm.insertvalue %[[S7]], %[[S6]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S9:.+]] = llvm.insertvalue %[[S3]], %[[S8]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S10:.+]] = llvm.insertvalue %[[S1]], %[[S9]][3, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S11:.+]] = llvm.insertvalue %[[S2]], %[[S10]][4, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S12:.+]] = llvm.insertvalue %[[S0]], %[[S11]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S13:.+]] = llvm.insertvalue %[[S1]], %[[S12]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S14:.+]] = builtin.unrealized_conversion_cast %13 : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32, #gpu.address_space<workgroup>>
// CHECK: "test.use.shared.memory"(%[[S14]]) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()

    gpu.return
  }

// CHECK-LABEL: llvm.func @func_device_function
  func.func @func_device_function()  {    
    %c8192 = arith.constant 8192 : index
    %shmem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %0 = memref.view %shmem[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32x64xf32, #gpu.address_space<workgroup>>
    "test.use.shared.memory"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
// CHECK: %[[S0:.+]] = llvm.mlir.constant(32 : index) : i64
// CHECK: %[[S1:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK: %[[S2:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[S3:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[S4:.+]] = llvm.mlir.addressof @__dynamic_shmem__3 : !llvm.ptr<3>
// CHECK: %[[S5:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[S6:.+]] = llvm.insertvalue %[[S4]], %[[S5]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S7:.+]] = llvm.getelementptr %[[S4]][8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: %[[S8:.+]] = llvm.insertvalue %[[S7]], %[[S6]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S9:.+]] = llvm.insertvalue %[[S3]], %[[S8]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S10:.+]] = llvm.insertvalue %[[S1]], %[[S9]][3, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S11:.+]] = llvm.insertvalue %[[S2]], %[[S10]][4, 1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S12:.+]] = llvm.insertvalue %[[S0]], %[[S11]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S13:.+]] = llvm.insertvalue %[[S1]], %[[S12]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
// CHECK: %[[S14:.+]] = builtin.unrealized_conversion_cast %13 : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32, #gpu.address_space<workgroup>>
// CHECK: "test.use.shared.memory"(%[[S14]]) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()

    func.return
  }
}
