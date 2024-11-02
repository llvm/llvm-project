; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; DISABLED-CHECK-DAG: OpName [[FNEG:%.+]] "scalar_fneg"
; CHECK-DAG: OpName [[FADD:%.+]] "test_fadd"
; CHECK-DAG: OpName [[FSUB:%.+]] "test_fsub"
; CHECK-DAG: OpName [[FMUL:%.+]] "test_fmul"
; CHECK-DAG: OpName [[FDIV:%.+]] "test_fdiv"
; CHECK-DAG: OpName [[FREM:%.+]] "test_frem"
; CHECK-DAG: OpName [[FMA:%.+]] "test_fma"

; CHECK-DAG: [[F32Ty:%.+]] = OpTypeFloat 32
; CHECK-DAG: [[FNTy:%.+]] = OpTypeFunction [[F32Ty]] [[F32Ty]] [[F32Ty]]


; CHECK:      [[FADD]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[C:%.+]] = OpFAdd [[F32Ty]] [[A]] [[B]]
;; TODO: OpDecorate checks
; CHECK-NEXT: OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fadd(float %a, float %b) {
    %c = fadd nnan ninf float %a, %b
    ret float %c
}

; CHECK:      [[FSUB]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[C:%.+]] = OpFSub [[F32Ty]] [[A]] [[B]]
;; TODO: OpDecorate checks
; CHECK-NEXT: OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fsub(float %a, float %b) {
    %c = fsub fast float %a, %b
    ret float %c
}

; CHECK:      [[FMUL]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[C:%.+]] = OpFMul [[F32Ty]] [[A]] [[B]]
;; TODO: OpDecorate checks]
; CHECK-NEXT: OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fmul(float %a, float %b) {
    %c = fmul contract float %a, %b
    ret float %c
}

; CHECK:      [[FDIV]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[C:%.+]] = OpFDiv [[F32Ty]] [[A]] [[B]]
;; TODO: OpDecorate checks
; CHECK-NEXT: OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fdiv(float %a, float %b) {
    %c = fdiv arcp nsz float %a, %b
    ret float %c
}

; CHECK:      [[FREM]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[C:%.+]] = OpFRem [[F32Ty]] [[A]] [[B]]
;; TODO: OpDecorate checks
; CHECK-NEXT: OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
define float @test_frem(float %a, float %b) {
    %c = frem nsz float %a, %b
    ret float %c
}


declare float @llvm.fma.f32(float, float, float)

; CHECK:      [[FMA]] = OpFunction
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.+]] = OpExtInst [[F32Ty]] {{%.+}} fma [[A]] [[B]] [[C]]
;; TODO: OpDecorate checks
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fma(float %a, float %b, float %c) {
    %r = call float @llvm.fma.f32(float %a, float %b, float %c)
    ret float %r
}
