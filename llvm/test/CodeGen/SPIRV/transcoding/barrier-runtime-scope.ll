; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
 
; CHECK: %[[#SCOPE:]] = OpFunctionCall %[[#]] %[[#]]
; CHECK: OpControlBarrier %[[#]] %[[#SCOPE]] %[[#]]
 
define spir_func void @_Z3foov() {
  %1 = call noundef i32 @_Z8getScopev()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef %1, i32 noundef 912)
  ret void
}
 
declare spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)
 
declare spir_func i32 @_Z8getScopev()
