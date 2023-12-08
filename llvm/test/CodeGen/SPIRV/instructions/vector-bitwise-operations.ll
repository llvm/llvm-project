; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[VECTOR_SHL:%.+]] "vector_shl"
; CHECK-DAG: OpName [[VECTOR_LSHR:%.+]] "vector_lshr"
; CHECK-DAG: OpName [[VECTOR_ASHR:%.+]] "vector_ashr"
; CHECK-DAG: OpName [[VECTOR_AND:%.+]] "vector_and"
; CHECK-DAG: OpName [[VECTOR_OR:%.+]] "vector_or"
; CHECK-DAG: OpName [[VECTOR_XOR:%.+]] "vector_xor"

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[I16:%.+]] = OpTypeInt 16
; CHECK-DAG: [[VECTOR:%.+]] = OpTypeVector [[I16]]
; CHECK-DAG: [[VECTOR_FN:%.+]] = OpTypeFunction [[VECTOR]] [[VECTOR]] [[VECTOR]]

; CHECK-NOT: DAG-FENCE


;; Test shl on vector:
define <2 x i16> @vector_shl(<2 x i16> %a, <2 x i16> %b) {
    %c = shl <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SHL]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpShiftLeftLogical [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test lshr on vector:
define <2 x i16> @vector_lshr(<2 x i16> %a, <2 x i16> %b) {
    %c = lshr <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_LSHR]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpShiftRightLogical [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test ashr on vector:
define <2 x i16> @vector_ashr(<2 x i16> %a, <2 x i16> %b) {
    %c = ashr <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_ASHR]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpShiftRightArithmetic [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test and on vector:
define <2 x i16> @vector_and(<2 x i16> %a, <2 x i16> %b) {
    %c = and <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_AND]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpBitwiseAnd [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test or on vector:
define <2 x i16> @vector_or(<2 x i16> %a, <2 x i16> %b) {
    %c = or <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_OR]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpBitwiseOr [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test xor on vector:
define <2 x i16> @vector_xor(<2 x i16> %a, <2 x i16> %b) {
    %c = xor <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_XOR]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpBitwiseXor [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
