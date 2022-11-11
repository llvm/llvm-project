; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; Types:
; CHECK-DAG:  %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG:  %[[#FNF32:]] = OpTypeFunction %[[#F32]] %[[#F32]]
;; Function decl:
; CHECK:      %[[#ANON:]] = OpFunction %[[#F32]] None %[[#FNF32]]
; CHECK-NEXT: OpFunctionParameter %[[#F32]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd
define internal spir_func float @0(float %a) {
  ret float %a
}

; CHECK:      OpFunctionCall %[[#F32]] %[[#ANON]]
define spir_kernel void @foo(float %a) {
  %1 = call spir_func float @0(float %a)
  ret void
}
