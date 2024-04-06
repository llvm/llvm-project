; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450
; CHECK: OpName %[[#all_bool_arg:]] "a"
; CHECK: OpName %[[#all_bool_ret:]] "hlsl.all"
; CHECK: %[[#bool:]] = OpTypeBool

define noundef i1 @all_bool(i1 noundef %a) {
entry:
  ; CHECK: %[[#all_bool_arg:]] = OpFunctionParameter %[[#bool:]]
  ; CHECK: OpLoad %[[#bool:]] %[[#all_bool_arg:]]
  ; CHECK: OpStore %[[#all_bool_ret:]] %[[#all_bool_arg:]]
  ; CHECK: OpReturnValue %[[#all_bool_ret:]]
  %hlsl.all = call i1 @llvm.spv.all.i1(i1 %a)
  ret i1 %hlsl.all
}