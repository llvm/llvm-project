// RUN: mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.EmitMeshTasksEXT
//===----------------------------------------------------------------------===//

func.func @emit_mesh_tasks(%0 : i32) {
  // CHECK: spirv.EXT.EmitMeshTasks {{%.*}}, {{%.*}}, {{%.*}} : i32, i32, i32
  spirv.EXT.EmitMeshTasks %0, %0, %0 : i32, i32, i32
}

func.func @emit_mesh_tasks_payload(%0 : i32, %1 : !spirv.ptr<i32, TaskPayloadWorkgroupEXT>) {
  // CHECK: spirv.EXT.EmitMeshTasks {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : i32, i32, i32, !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
  spirv.EXT.EmitMeshTasks %0, %0, %0, %1 : i32, i32, i32, !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
}

// -----

func.func @emit_mesh_tasks_wrong_payload(%0 : i32, %1 : !spirv.ptr<i32, Image>) {
  // expected-error @+1 {{payload must be a variable with a storage class of TaskPayloadWorkgroupEXT}}
  spirv.EXT.EmitMeshTasks %0, %0, %0, %1 : i32, i32, i32, !spirv.ptr<i32, Image>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SetMeshOutputsEXT
//===----------------------------------------------------------------------===//

func.func @set_mesh_outputs(%0 : i32, %1 : i32) {
  // CHECK: spirv.EXT.SetMeshOutputs {{%.*}}, {{%.*}} : i32, i32
  spirv.EXT.SetMeshOutputs %0, %1 : i32, i32
  spirv.Return
}
