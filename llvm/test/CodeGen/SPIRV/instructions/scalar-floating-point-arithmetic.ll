; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; DISABLED-CHECK-DAG: OpName [[SCALAR_FNEG:%.+]] "scalar_fneg"
; CHECK-DAG: OpName [[SCALAR_FADD:%.+]] "scalar_fadd"
; CHECK-DAG: OpName [[SCALAR_FSUB:%.+]] "scalar_fsub"
; CHECK-DAG: OpName [[SCALAR_FMUL:%.+]] "scalar_fmul"
; CHECK-DAG: OpName [[SCALAR_FDIV:%.+]] "scalar_fdiv"
; CHECK-DAG: OpName [[SCALAR_FREM:%.+]] "scalar_frem"
; CHECK-DAG: OpName [[SCALAR_FMA:%.+]] "scalar_fma"
;; FIXME: add test for OpFMod

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[SCALAR:%.+]] = OpTypeFloat 32
; CHECK-DAG: [[SCALAR_FN:%.+]] = OpTypeFunction [[SCALAR]] [[SCALAR]] [[SCALAR]]

; CHECK-NOT: DAG-FENCE


;; Test fneg on scalar:
;; FIXME: Uncomment this test once we have rebased onto a more recent LLVM
;;        version -- IRTranslator::translateFNeg was fixed.
;; define float @scalar_fneg(float %a, float %unused) {
;;     %c = fneg float %a
;;     ret float %c
;; }

; DISABLED-CHECK:      [[SCALAR_FNEG]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; DISABLED-CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; DISABLED-CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; DISABLED-CHECK:      OpLabel
; DISABLED-CHECK:      [[C:%.+]] = OpFNegate [[SCALAR]] [[A]]
; DISABLED-CHECK:      OpReturnValue [[C]]
; DISABLED-CHECK-NEXT: OpFunctionEnd


;; Test fadd on scalar:
define float @scalar_fadd(float %a, float %b) {
    %c = fadd float %a, %b
    ret float %c
}

; CHECK:      [[SCALAR_FADD]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFAdd [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fsub on scalar:
define float @scalar_fsub(float %a, float %b) {
    %c = fsub float %a, %b
    ret float %c
}

; CHECK:      [[SCALAR_FSUB]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFSub [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fmul on scalar:
define float @scalar_fmul(float %a, float %b) {
    %c = fmul float %a, %b
    ret float %c
}

; CHECK:      [[SCALAR_FMUL]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFMul [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test fdiv on scalar:
define float @scalar_fdiv(float %a, float %b) {
    %c = fdiv float %a, %b
    ret float %c
}

; CHECK:      [[SCALAR_FDIV]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFDiv [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test frem on scalar:
define float @scalar_frem(float %a, float %b) {
    %c = frem float %a, %b
    ret float %c
}

; CHECK:      [[SCALAR_FREM]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpFRem [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd

declare float @llvm.fma.f32(float, float, float)

;; Test fma on scalar:
define float @scalar_fma(float %a, float %b, float %c) {
    %r = call float @llvm.fma.f32(float %a, float %b, float %c)
    ret float %r
}

; CHECK:      [[SCALAR_FMA]] = OpFunction
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[R:%.+]] = OpExtInst [[SCALAR]] {{%.+}} fma [[A]] [[B]] [[C]]
; CHECK:      OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
