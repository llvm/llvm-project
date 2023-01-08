; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; Types:
; CHECK-DAG:  %[[#I32:]] = OpTypeInt 32
; CHECK-DAG:  %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG:  %[[#FNI32:]] = OpTypeFunction %[[#I32]] %[[#I32]]
; CHECK-DAG:  %[[#FNF32:]] = OpTypeFunction %[[#F32]] %[[#F32]]

;; Function declarations:
; CHECK:      %[[#ANON0:]] = OpFunction %[[#I32]] None %[[#FNI32]]
; CHECK-NEXT: OpFunctionParameter %[[#I32]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd
define internal spir_func i32 @0(i32 %a) {
  ret i32 %a
}

; CHECK:      %[[#ANON1:]] = OpFunction %[[#F32]] None %[[#FNF32]]
; CHECK-NEXT: OpFunctionParameter %[[#F32]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd
define internal spir_func float @1(float %a) {
  ret float %a
}

;; Calls:
; CHECK:      OpFunctionCall %[[#I32]] %[[#ANON0]]
; CHECK:      OpFunctionCall %[[#F32]] %[[#ANON1]]
define spir_kernel void @foo(i32 %a) {
  %call1 = call spir_func i32 @0(i32 %a)
  %b = sitofp i32 %a to float
  %call2 = call spir_func float @1(float %b)
  ret void
}
