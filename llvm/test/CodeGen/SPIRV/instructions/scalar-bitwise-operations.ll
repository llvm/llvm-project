; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[SCALAR_SHL:%.+]] "scalar_shl"
; CHECK-DAG: OpName [[SCALAR_LSHR:%.+]] "scalar_lshr"
; CHECK-DAG: OpName [[SCALAR_ASHR:%.+]] "scalar_ashr"
; CHECK-DAG: OpName [[SCALAR_AND:%.+]] "scalar_and"
; CHECK-DAG: OpName [[SCALAR_OR:%.+]] "scalar_or"
; CHECK-DAG: OpName [[SCALAR_XOR:%.+]] "scalar_xor"

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[SCALAR:%.+]] = OpTypeInt 32
; CHECK-DAG: [[SCALAR_FN:%.+]] = OpTypeFunction [[SCALAR]] [[SCALAR]] [[SCALAR]]

; CHECK-NOT: DAG-FENCE


;; Test shl on scalar:
define i32 @scalar_shl(i32 %a, i32 %b) {
    %c = shl i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_SHL]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpShiftLeftLogical [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test lshr on scalar:
define i32 @scalar_lshr(i32 %a, i32 %b) {
    %c = lshr i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_LSHR]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpShiftRightLogical [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test ashr on scalar:
define i32 @scalar_ashr(i32 %a, i32 %b) {
    %c = ashr i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_ASHR]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpShiftRightArithmetic [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test and on scalar:
define i32 @scalar_and(i32 %a, i32 %b) {
    %c = and i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_AND]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpBitwiseAnd [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test or on scalar:
define i32 @scalar_or(i32 %a, i32 %b) {
    %c = or i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_OR]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpBitwiseOr [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test xor on scalar:
define i32 @scalar_xor(i32 %a, i32 %b) {
    %c = xor i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_XOR]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpBitwiseXor [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
