; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[VECTOR_FNEG:%.+]] "vector_fneg"
; CHECK-DAG: OpName [[VECTOR_FADD:%.+]] "vector_fadd"
; CHECK-DAG: OpName [[VECTOR_FSUB:%.+]] "vector_fsub"
; CHECK-DAG: OpName [[VECTOR_FMUL:%.+]] "vector_fmul"
; CHECK-DAG: OpName [[VECTOR_FDIV:%.+]] "vector_fdiv"
; CHECK-DAG: OpName [[VECTOR_FREM:%.+]] "vector_frem"
;; TODO: add test for OpFMod

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[FP16:%.+]] = OpTypeFloat 16
; CHECK-DAG: [[VECTOR:%.+]] = OpTypeVector [[FP16]]
; CHECK-DAG: [[VECTOR_FN:%.+]] = OpTypeFunction [[VECTOR]] [[VECTOR]] [[VECTOR]]

; CHECK-NOT: DAG-FENCE


;; Test fneg on vector:
define <2 x half> @vector_fneg(<2 x half> %a, <2 x half> %unused) {
    %c = fneg <2 x half> %a
    ret <2 x half> %c
}

; CHECK:      [[VECTOR_FNEG]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFNegate [[VECTOR]] [[A]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fadd on vector:
define <2 x half> @vector_fadd(<2 x half> %a, <2 x half> %b) {
    %c = fadd <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK:      [[VECTOR_FADD]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFAdd [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fsub on vector:
define <2 x half> @vector_fsub(<2 x half> %a, <2 x half> %b) {
    %c = fsub <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK:      [[VECTOR_FSUB]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFSub [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fmul on vector:
define <2 x half> @vector_fmul(<2 x half> %a, <2 x half> %b) {
    %c = fmul <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK:      [[VECTOR_FMUL]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFMul [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fdiv on vector:
define <2 x half> @vector_fdiv(<2 x half> %a, <2 x half> %b) {
    %c = fdiv <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK:      [[VECTOR_FDIV]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFDiv [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test frem on vector:
define <2 x half> @vector_frem(<2 x half> %a, <2 x half> %b) {
    %c = frem <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK:      [[VECTOR_FREM]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFRem [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
