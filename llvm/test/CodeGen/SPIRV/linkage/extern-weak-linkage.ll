; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: Capability Linkage
; CHECK-SPIRV-DAG: OpName %[[#AbsFun:]] "abs"
; CHECK-SPIRV-DAG: OpName %[[#ExternalFun:]] "__devicelib_abs"
; CHECK-SPIRV-DAG: OpDecorate %[[#AbsFun]] LinkageAttributes "abs" Export
; CHECK-SPIRV-DAG: OpDecorate %[[#ExternalFun]] LinkageAttributes "__devicelib_abs" Import

define weak dso_local spir_func i32 @abs(i32 noundef %x) {
entry:
  %call = tail call spir_func i32 @__devicelib_abs(i32 noundef %x) #11
  ret i32 %call
}

declare extern_weak dso_local spir_func i32 @__devicelib_abs(i32 noundef)
