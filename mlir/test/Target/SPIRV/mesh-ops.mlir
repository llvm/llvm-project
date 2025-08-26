// RUN: mlir-translate --no-implicit-module --split-input-file --test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.4, [MeshShadingEXT], [SPV_EXT_mesh_shader]> {
  // CHECK-LABEL: @emit_mesh_tasks
  spirv.func @emit_mesh_tasks() "None" {
    %0 = spirv.Constant 1 : i32
    // CHECK: spirv.EXT.EmitMeshTasks {{%.*}}, {{%.*}}, {{%.*}} : i32, i32, i32
    spirv.EXT.EmitMeshTasks %0, %0, %0 : i32, i32, i32
  }
  // CHECK-LABEL: @set_mesh_outputs
  spirv.func @set_mesh_outputs(%0 : i32, %1 : i32) "None" {
    // CHECK: spirv.EXT.SetMeshOutputs {{%.*}}, {{%.*}} : i32, i32
    spirv.EXT.SetMeshOutputs %0, %1 : i32, i32
    spirv.Return
  }
  // CHECK: spirv.EntryPoint "TaskEXT" {{@.*}}
  spirv.EntryPoint "TaskEXT" @emit_mesh_tasks
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.4, [MeshShadingEXT], [SPV_EXT_mesh_shader]> {
  spirv.GlobalVariable @payload : !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
  // CHECK-LABEL: @emit_mesh_tasks_payload
  spirv.func @emit_mesh_tasks_payload() "None" {
    %0 = spirv.Constant 1 : i32
    %1 = spirv.mlir.addressof @payload : !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
    // CHECK: spirv.EXT.EmitMeshTasks {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : i32, i32, i32, !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
    spirv.EXT.EmitMeshTasks %0, %0, %0, %1 : i32, i32, i32, !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
  }
  // CHECK: spirv.EntryPoint "TaskEXT" {{@.*}}, {{@.*}}
  spirv.EntryPoint "TaskEXT" @emit_mesh_tasks_payload, @payload
}
