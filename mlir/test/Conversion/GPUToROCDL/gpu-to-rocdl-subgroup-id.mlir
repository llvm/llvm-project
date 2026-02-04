// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx942' | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1201' | FileCheck %s --check-prefixes=CHECK,GFX12

gpu.module @test_module {
// CHECK-LABEL: func @subgroup_id()
func.func @subgroup_id() -> index {
  // GFX12: rocdl.wave.id : i32
  // GFX12: llvm.sext %{{.*}} : i32 to i64

  // GFX9-DAG: rocdl.workitem.id.x : i32
  // GFX9-DAG: rocdl.workitem.id.y : i32
  // GFX9-DAG: rocdl.workitem.id.z : i32
  // GFX9-DAG: rocdl.workgroup.dim.x : i32
  // GFX9-DAG: rocdl.workgroup.dim.y : i32
  // GFX9-DAG: llvm.mul %{{.*}}, %{{.*}} overflow<nsw, nuw>
  // GFX9-DAG: llvm.add %{{.*}}, %{{.*}} overflow<nsw, nuw>
  // GFX9: rocdl.wavefrontsize : i32
  // GFX9: llvm.udiv
  // GFX9: llvm.sext %{{.*}} : i32 to i64
  %subgroupId = gpu.subgroup_id : index
  func.return %subgroupId : index
}

// CHECK-LABEL: func @subgroup_id_with_upper_bound()
func.func @subgroup_id_with_upper_bound() -> index {
  // GFX12: rocdl.wave.id range <i32, 0, 4> : i32
  // GFX12: llvm.sext %{{.*}} : i32 to i64

  // GFX9-DAG: rocdl.workitem.id.x : i32
  // GFX9-DAG: rocdl.workitem.id.y : i32
  // GFX9-DAG: rocdl.workitem.id.z : i32
  // GFX9-DAG: rocdl.workgroup.dim.x : i32
  // GFX9-DAG: rocdl.workgroup.dim.y : i32
  // GFX9: rocdl.wavefrontsize : i32
  // GFX9: llvm.udiv
  // GFX9: llvm.sext %{{.*}} : i32 to i64
  %subgroupId = gpu.subgroup_id upper_bound 4 : index
  func.return %subgroupId : index
}
}
