// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file | FileCheck %s

gpu.module @test_module {
// CHECK-LABEL: func @subgroup_id()
func.func @subgroup_id() -> index {
  // CHECK-DAG: = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK-DAG: = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK-DAG: = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK-DAG: = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK-DAG: = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: = nvvm.read.ptx.sreg.warpsize range <i32, 32, 33> : i32
  // CHECK: = llvm.udiv %{{.*}}, %{{.*}} : i64
  %subgroupId = gpu.subgroup_id : index
  func.return %subgroupId : index
}
}

// -----

// The block sizes are known here, so the linearization folds them to constants
// and the thread ids come out carrying the matching ranges.
gpu.module @test_module {
// CHECK-LABEL: func @subgroup_id_with_workgroup_sizes()
func.func @subgroup_id_with_workgroup_sizes() -> index
    attributes {gpu.known_block_size = array<i32: 32, 4, 2>} {
  // CHECK-DAG: = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
  // CHECK-DAG: = nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
  // CHECK-DAG: = nvvm.read.ptx.sreg.tid.z range <i32, 0, 2> : i32
  // CHECK: = nvvm.read.ptx.sreg.warpsize range <i32, 32, 33> : i32
  // CHECK: = llvm.udiv %{{.*}}, %{{.*}} : i64
  %subgroupId = gpu.subgroup_id : index
  func.return %subgroupId : index
}
}
