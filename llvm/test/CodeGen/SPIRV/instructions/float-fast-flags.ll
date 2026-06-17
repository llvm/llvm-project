; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; DISABLED-CHECK-DAG: OpName [[FNEG:%.+]] "scalar_fneg"
; CHECK-DAG: OpName [[FADD:%.+]] "test_fadd"
; CHECK-DAG: OpName [[FSUB:%.+]] "test_fsub"
; CHECK-DAG: OpName [[FMUL:%.+]] "test_fmul"
; CHECK-DAG: OpName [[FDIV:%.+]] "test_fdiv"
; CHECK-DAG: OpName [[FREM:%.+]] "test_frem"
; CHECK-DAG: OpName [[FMA:%.+]] "test_fma"

; CHECK: OpDecorate [[FADD]] LinkageAttributes "test_fadd" Export
; CHECK-NEXT: OpDecorate [[FADD_RES:%.+]] FPFastMathMode NotNaN|NotInf{{$}}
; CHECK: OpDecorate [[FSUB]] LinkageAttributes "test_fsub" Export
; CHECK-NEXT: OpDecorate [[FSUB_RES:%.+]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast{{$}}
; CHECK: OpDecorate [[FMUL]] LinkageAttributes "test_fmul" Export
; CHECK-NOT: FPFastMathMode
; CHECK: OpDecorate [[FDIV]] LinkageAttributes "test_fdiv" Export
; CHECK-NEXT: OpDecorate [[FDIV_RES:%.+]] FPFastMathMode NSZ|AllowRecip{{$}}
; CHECK: OpDecorate [[FREM]] LinkageAttributes "test_frem" Export
; CHECK-NEXT: OpDecorate [[FREM_RES:%.+]] FPFastMathMode NSZ{{$}}
; CHECK: OpDecorate [[FMA]] LinkageAttributes "test_fma" Export
; CHECK-NEXT: OpDecorate [[FMA_RES:%.+]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast{{$}}
; CHECK: [[F32Ty:%.+]] = OpTypeFloat 32
; CHECK-DAG: [[FNTy:%.+]] = OpTypeFunction [[F32Ty]] [[F32Ty]] [[F32Ty]]


; CHECK:      [[FADD]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[FADD_RES]] = OpFAdd [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[FADD_RES]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fadd(float %a, float %b) {
    %c = fadd nnan ninf float %a, %b
    ret float %c
}

; CHECK:      [[FSUB]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[FSUB_RES]] = OpFSub [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[FSUB_RES]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fsub(float %a, float %b) {
    %c = fsub fast float %a, %b
    ret float %c
}

; CHECK:      [[FMUL]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[FMUL_RES:%.+]] = OpFMul [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[FMUL_RES]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fmul(float %a, float %b) {
    %c = fmul contract float %a, %b
    ret float %c
}

; CHECK:      [[FDIV]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[FDIV_RES]] = OpFDiv [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[FDIV_RES]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fdiv(float %a, float %b) {
    %c = fdiv arcp nsz float %a, %b
    ret float %c
}

; CHECK:      [[FREM]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[FREM_RES]] = OpFRem [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[FREM_RES]]
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
; CHECK-NEXT: [[FMA_RES]] = OpExtInst [[F32Ty]] {{%.+}} fma [[A]] [[B]] [[C]]
; CHECK-NEXT: OpReturnValue [[FMA_RES]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fma(float %a, float %b, float %c) {
    %r = call fast float @llvm.fma.f32(float %a, float %b, float %c)
    ret float %r
}
