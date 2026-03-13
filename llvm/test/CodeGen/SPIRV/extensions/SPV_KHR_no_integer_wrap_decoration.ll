; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_no_integer_wrap_decoration %s -o - | FileCheck %s

; CHECK-DAG: OpExtension "SPV_KHR_no_integer_wrap_decoration"

; CHECK-NOT: DAG-FENCE

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: OpDecorate %[[#C:]] NoUnsignedWrap
; CHECK-DAG: OpDecorate %[[#D:]] NoSignedWrap
; CHECK-DAG: OpDecorate %[[#E:]] NoUnsignedWrap
; CHECK-DAG: OpDecorate %[[#E]] NoSignedWrap

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32
; CHECK-DAG: %[[#FN:]] = OpTypeFunction %[[#I32]] %[[#I32]] %[[#I32]]

; CHECK-NOT: DAG-FENCE

define i32 @no_wrap_test(i32 %a, i32 %b) {
    %c = mul nuw i32 %a, %b
    %d = mul nsw i32 %a, %b
    %e = add nuw nsw i32 %c, %d
    ret i32 %e
}

; CHECK:      OpFunction %[[#I32]] None %[[#FN]]
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter %[[#I32]]
; CHECK-NEXT: %[[#B:]] = OpFunctionParameter %[[#I32]]
; CHECK:      OpLabel
; CHECK:      %[[#C]] = OpIMul %[[#I32]] %[[#A]] %[[#B]]
; CHECK:      %[[#D]] = OpIMul %[[#I32]] %[[#A]] %[[#B]]
; CHECK:      %[[#E]] = OpIAdd %[[#I32]] %[[#C]] %[[#D]]
; CHECK:      OpReturnValue %[[#E]]
; CHECK-NEXT: OpFunctionEnd
