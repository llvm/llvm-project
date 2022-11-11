; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[VECTOR_ADD:%.+]] "vector_add"
; CHECK-DAG: OpName [[VECTOR_SUB:%.+]] "vector_sub"
; CHECK-DAG: OpName [[VECTOR_MUL:%.+]] "vector_mul"
; CHECK-DAG: OpName [[VECTOR_UDIV:%.+]] "vector_udiv"
; CHECK-DAG: OpName [[VECTOR_SDIV:%.+]] "vector_sdiv"
;; TODO: add tests for urem + srem
;; TODO: add test for OpSNegate

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[I16:%.+]] = OpTypeInt 16
; CHECK-DAG: [[VECTOR:%.+]] = OpTypeVector [[I16]]
; CHECK-DAG: [[VECTOR_FN:%.+]] = OpTypeFunction [[VECTOR]] [[VECTOR]] [[VECTOR]]

; CHECK-NOT: DAG-FENCE


;; Test add on vector:
define <2 x i16> @vector_add(<2 x i16> %a, <2 x i16> %b) {
    %c = add <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_ADD]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpIAdd [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test sub on vector:
define <2 x i16> @vector_sub(<2 x i16> %a, <2 x i16> %b) {
    %c = sub <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SUB]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpISub [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test mul on vector:
define <2 x i16> @vector_mul(<2 x i16> %a, <2 x i16> %b) {
    %c = mul <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_MUL]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpIMul [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test udiv on vector:
define <2 x i16> @vector_udiv(<2 x i16> %a, <2 x i16> %b) {
    %c = udiv <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_UDIV]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpUDiv [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test sdiv on vector:
define <2 x i16> @vector_sdiv(<2 x i16> %a, <2 x i16> %b) {
    %c = sdiv <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SDIV]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpSDiv [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
