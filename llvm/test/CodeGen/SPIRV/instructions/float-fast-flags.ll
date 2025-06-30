; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s

; DISABLED-CHECK-DAG: OpName [[FNEG:%.+]] "scalar_fneg"
; CHECK-DAG: OpName [[FADD:%.+]] "test_fadd"
; CHECK-DAG: OpName [[FSUB:%.+]] "test_fsub"
; CHECK-DAG: OpName [[FMUL:%.+]] "test_fmul"
; CHECK-DAG: OpName [[FDIV:%.+]] "test_fdiv"
; CHECK-DAG: OpName [[FREM:%.+]] "test_frem"
; CHECK-DAG: OpName [[FMA:%.+]] "test_fma"
; CHECK-DAG: OpDecorate %[[#FAddC:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#FSubC:]] FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform
; CHECK-DAG: OpDecorate %[[#FMulC:]] FPFastMathMode AllowContract
; CHECK-DAG: OpDecorate %[[#FDivC:]] FPFastMathMode NSZ|AllowRecip
; CHECK-DAG: OpDecorate %[[#FRemC:]] FPFastMathMode NSZ

; CHECK-DAG: [[F32Ty:%.+]] = OpTypeFloat 32
; CHECK-DAG: [[FNTy:%.+]] = OpTypeFunction [[F32Ty]] [[F32Ty]] [[F32Ty]]


; CHECK:      [[FADD]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#FAddC]] = OpFAdd [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue %[[#FAddC]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fadd(float %a, float %b) {
    %c = fadd nnan ninf float %a, %b
    ret float %c
}

; CHECK:      [[FSUB]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#FSubC]] = OpFSub [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue %[[#FSubC]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fsub(float %a, float %b) {
    %c = fsub fast float %a, %b
    ret float %c
}

; CHECK:      [[FMUL]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#FMulC]] = OpFMul [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue %[[#FMulC]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fmul(float %a, float %b) {
    %c = fmul contract float %a, %b
    ret float %c
}

; CHECK:      [[FDIV]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#FDivC]] = OpFDiv [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue %[[#FDivC]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fdiv(float %a, float %b) {
    %c = fdiv arcp nsz float %a, %b
    ret float %c
}

; CHECK:      [[FREM]] = OpFunction [[F32Ty]] None [[FNTy]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[F32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#FRemC]] = OpFRem [[F32Ty]] [[A]] [[B]]
; CHECK-NEXT: OpReturnValue %[[#FRemC]]
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
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define float @test_fma(float %a, float %b, float %c) {
    %r = call float @llvm.fma.f32(float %a, float %b, float %c)
    ret float %r
}
