; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that duplicate align information does not result in SPIR-V validation
; errors due to duplicate Alignment Decorations.

;CHECK: OpDecorate %[[#Var:]] Alignment
;CHECK: %[[#Var]] = OpVariable %[[#]]

define spir_func void @f() {
 %res = alloca i16, align 2, !spirv.Decorations !1
 ret void
}

!1 = !{!2}
!2 = !{i32 44, i32 2}
