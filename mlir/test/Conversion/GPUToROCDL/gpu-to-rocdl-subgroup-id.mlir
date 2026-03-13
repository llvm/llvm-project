// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx942' | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1201' | FileCheck %s --check-prefixes=CHECK,GFX12

gpu.module @test_module {
// CHECK-LABEL: func @subgroup_id()
func.func @subgroup_id() -> index {
  // GFX12: %[[WAVEID:.+]] = rocdl.wave.id : i32
  // GFX12: llvm.sext %[[WAVEID]] : i32 to i64

  // GFX9-DAG: %[[IDX:.+]] = rocdl.workitem.id.x : i32
  // GFX9-DAG: %[[IDY:.+]] = rocdl.workitem.id.y : i32
  // GFX9-DAG: %[[IDZ:.+]] = rocdl.workitem.id.z : i32
  // GFX9-DAG: %[[DIMX_I64:.+]] = llvm.call @__ockl_get_local_size(%[[C0:.+]]) : (i32) -> (i64 {llvm.range = #llvm.constant_range<i64, 1, 1025>})
  // Yes, this is checking after the call that uses it. This prevents collisions with other 0s.
  // GFX9-DAG: %[[C0]] = llvm.mlir.constant(0 : i32) : i32
  // GFX9-DAG: %[[DIMX:.+]] = llvm.trunc %[[DIMX_I64]] overflow<nsw, nuw> : i64 to i32
  // GFX9-DAG: %[[DIMY_I64:.+]] = llvm.call @__ockl_get_local_size(%[[C1:.+]]) : (i32) -> (i64 {llvm.range = #llvm.constant_range<i64, 1, 1025>})
  // GFX9-DAG: %[[C1]] = llvm.mlir.constant(1 : i32) : i32
  // GFX9-DAG: %[[DIMY:.+]] = llvm.trunc %[[DIMY_I64]] overflow<nsw, nuw> : i64 to i32
  // GFX9: %[[Z_DY:.+]] = llvm.mul %[[DIMY]], %[[IDZ]] overflow<nsw, nuw>
  // GFX9: %[[ZY:.+]] = llvm.add %[[IDY]], %[[Z_DY]] overflow<nsw, nuw>
  // GFX9: %[[YZ_DX:.+]] = llvm.mul %[[DIMX]], %[[ZY]] overflow<nsw, nuw>
  // GFX9: %[[ZYX:.+]] = llvm.add %[[IDX]], %[[YZ_DX]] overflow<nsw, nuw>
  // GFX9: %[[WAVESZ:.+]] = rocdl.wavefrontsize : i32
  // GFX9: %[[RES:.+]] = llvm.udiv %[[ZYX]], %[[WAVESZ]]
  // GFX9: llvm.sext %[[RES]] : i32 to i64
  %subgroupId = gpu.subgroup_id : index
  func.return %subgroupId : index
}

// CHECK-LABEL: func @subgroup_id_with_upper_bound()
func.func @subgroup_id_with_upper_bound() -> index {
  // GFX12: %[[WAVEID:.+]] = rocdl.wave.id range <i32, 0, 4> : i32
  // GFX12: llvm.sext %[[WAVEID]] : i32 to i64

  // Minimal check to ensure we don't set any bounds based on the subgroup ID bound
  // since we don't know which thread ID they go on to.
  // GFX9: rocdl.workitem.id.x : i32
  // GFX9-DAG: llvm.call @__ockl_get_local_size({{.*}}) : (i32) -> (i64 {llvm.range = #llvm.constant_range<i64, 1, 1025>})
  %subgroupId = gpu.subgroup_id upper_bound 4 : index
  func.return %subgroupId : index
}

// CHECK-LABEL: func @subgroup_id_with_workgroup_sizes()
func.func @subgroup_id_with_workgroup_sizes() -> index
    attributes {gpu.known_block_size = array<i32: 64, 4, 1>} {
  // GFX12: %[[WAVEID:.+]] = rocdl.wave.id range <i32, 0, 4> : i32
  // GFX12: llvm.sext %[[WAVEID]] : i32 to i64

  // GFX9-DAG: %[[IDX:.+]] = rocdl.workitem.id.x range <i32, 0, 64> : i32
  // GFX9-DAG: %[[IDY:.+]] = rocdl.workitem.id.y range <i32, 0, 4> : i32
  // GFX9-DAG: %[[IDZ:.+]] = rocdl.workitem.id.z range <i32, 0, 1> : i32
  // GFX9-DAG: %[[DIMX_I64:.+]] = llvm.call @__ockl_get_local_size(%[[C0:.+]]) : (i32) -> (i64 {llvm.range = #llvm.constant_range<i64, 1, 65>})
  // Yes, this is checking after the call that uses it. This prevents collisions with other 0s.
  // GFX9-DAG: %[[C0]] = llvm.mlir.constant(0 : i32) : i32
  // GFX9-DAG: %[[DIMX:.+]] = llvm.trunc %[[DIMX_I64]] overflow<nsw, nuw> : i64 to i32
  // GFX9-DAG: %[[DIMY_I64:.+]] = llvm.call @__ockl_get_local_size(%[[C1:.+]]) : (i32) -> (i64 {llvm.range = #llvm.constant_range<i64, 1, 5>})
  // GFX9-DAG: %[[C1]] = llvm.mlir.constant(1 : i32) : i32
  // GFX9-DAG: %[[DIMY:.+]] = llvm.trunc %[[DIMY_I64]] overflow<nsw, nuw> : i64 to i32
  // GFX9: %[[Z_DY:.+]] = llvm.mul %[[DIMY]], %[[IDZ]] overflow<nsw, nuw>
  // GFX9: %[[ZY:.+]] = llvm.add %[[IDY]], %[[Z_DY]] overflow<nsw, nuw>
  // GFX9: %[[YZ_DX:.+]] = llvm.mul %[[DIMX]], %[[ZY]] overflow<nsw, nuw>
  // GFX9: %[[ZYX:.+]] = llvm.add %[[IDX]], %[[YZ_DX]] overflow<nsw, nuw>
  // GFX9: %[[WAVESZ:.+]] = rocdl.wavefrontsize : i32
  // GFX9: %[[RES:.+]] = llvm.udiv %[[ZYX]], %[[WAVESZ]]
  // GFX9: llvm.sext %[[RES]] : i32 to i64
  %subgroupId = gpu.subgroup_id upper_bound 4 : index
  func.return %subgroupId : index
}
}
