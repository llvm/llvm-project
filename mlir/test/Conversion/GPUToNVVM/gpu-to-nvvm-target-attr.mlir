// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(convert-to-llvm{dynamic=true}))" | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(convert-to-llvm{dynamic=true allow-pattern-rollback=0}))" | FileCheck %s

// CHECK-LABEL: gpu.module @nvvm_module
gpu.module @nvvm_module [#nvvm.target] {
  // CHECK-LABEL: llvm.func @kernel_0()
  func.func @kernel_0() -> index {
    // CHECK: = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = gpu.thread_id x
    // CHECK: = nvvm.read.ptx.sreg.laneid range <i32, 0, 32> : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %laneId = gpu.lane_id
    %sum = index.add %tIdX, %laneId
    func.return %sum : index
  }

// CHECK-LABEL: llvm.func @kernel_1
// CHECK: (%{{.*}}: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64)
// CHECK: attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} 
  gpu.func @kernel_1(%arg0 : memref<f32, #gpu.address_space<global>>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
    gpu.return
  }
}

// CHECK-LABEL: gpu.module @nvvm_module_2
gpu.module @nvvm_module_2 {
  // CHECK-LABEL: llvm.func @kernel_0()
  func.func @kernel_0() -> index {
    // CHECK: = gpu.thread_id x
    %tIdX = gpu.thread_id x
    // CHECK: = gpu.lane_id
    %laneId = gpu.lane_id
    %sum = index.add %tIdX, %laneId
    func.return %sum : index
  }

// CHECK-LABEL: gpu.func @kernel_1
// CHECK: (%{{.*}}: memref<f32, #gpu.address_space<global>>) kernel attributes {known_block_size = array<i32: 128, 1, 1>}
  gpu.func @kernel_1(%arg0 : memref<f32, #gpu.address_space<global>>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
    gpu.return
  }
}
