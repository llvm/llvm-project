; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=FuncParamAttr
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=FuncParamAttr
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata %s -o - | FileCheck %s --implicit-check-not=FuncParamAttr --implicit-check-not='OpString "spirv.Decorations"'
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; Function decorations apply to the OpFunction ID for both definitions and
; declarations. These functions have no LLVM return attributes, so the
; FuncParamAttr decorations must come from the metadata.
; Calling the declaration from two functions exercises deduplication: the
; implicit negative check rejects any extra FuncParamAttr decorations.

; CHECK-DAG: OpName %[[#Defined:]] "defined"
; CHECK-DAG: OpName %[[#Imported:]] "imported"
; CHECK: OpDecorate %[[#Defined]] FuncParamAttr Zext
; CHECK: OpDecorate %[[#Imported]] FuncParamAttr Sext
; CHECK: %[[#Imported]] = OpFunction
; CHECK-NEXT: OpFunctionEnd
; CHECK: %[[#Defined]] = OpFunction

define spir_func i8 @defined() !spirv.Decorations !0 {
  %result = call spir_func i8 @imported()
  ret i8 %result
}

declare !spirv.Decorations !2 spir_func i8 @imported()

define spir_kernel void @kernel() {
  %a = call spir_func i8 @defined()
  %b = call spir_func i8 @imported()
  ret void
}

!0 = !{!1}
!1 = !{i32 38, i32 0} ; FuncParamAttr Zext
!2 = !{!3}
!3 = !{i32 38, i32 1} ; FuncParamAttr Sext
