; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s --check-prefixes=CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-linux %s -o - --spirv-ext=+SPV_EXT_arithmetic_fence | FileCheck %s --check-prefixes=CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-NOEXT-NO: OpCapability ArithmeticFenceEXT
; CHECK-NOEXT-NO: OpExtension "SPV_EXT_arithmetic_fence"
; CHECK-NOEXT: OpFunction
; CHECK-NOEXT: OpFMul
; CHECK-NOEXT: OpFAdd
; CHECK-NOEXT-NO: OpArithmeticFenceEXT
; CHECK-NOEXT: OpFunction
; CHECK-NOEXT-NO: OpArithmeticFenceEXT
; CHECK-NOEXT: OpFunction
; CHECK-NOEXT-NO: OpArithmeticFenceEXT

; CHECK-EXT: OpCapability ArithmeticFenceEXT
; CHECK-EXT: OpExtension "SPV_EXT_arithmetic_fence"
; CHECK-EXT: OpFunction
; CHECK-EXT: [[R1:%.*]] = OpFMul [[I32Ty:%.*]] %[[#]] %[[#]]
; CHECK-EXT: [[R2:%.*]] = OpArithmeticFenceEXT [[I32Ty]] [[R1]]
; CHECK-EXT: %[[#]] = OpFAdd [[I32Ty]] [[R2]] %[[#]]
; CHECK-EXT: OpFunction
; CHECK-EXT: [[R3:%.*]] = OpFAdd [[I64Ty:%.*]] [[A1:%.*]] [[A1]]
; CHECK-EXT: [[R4:%.*]] = OpArithmeticFenceEXT [[I64Ty]] [[R3]]
; CHECK-EXT: [[R5:%.*]] = OpFAdd [[I64Ty]] [[A1]] [[A1]]
; CHECK-EXT: %[[#]] = OpFAdd [[I64Ty]] [[R4]] [[R5]]
; CHECK-EXT: OpFunction
; CHECK-EXT: [[R6:%.*]] = OpFAdd [[I32VecTy:%.*]] [[A2:%.*]] [[A2]]
; CHECK-EXT: [[R7:%.*]] = OpArithmeticFenceEXT [[I32VecTy]] [[R6]]
; CHECK-EXT: [[R8:%.*]] = OpFAdd [[I32VecTy]] [[A2]] [[A2]]
; CHECK-EXT: %[[#]] = OpFAdd [[I32VecTy]] [[R7]] [[R8]]

define float @f1(float %a, float %b, float %c) {
  %mul = fmul fast float %b, %a
  %tmp = call float @llvm.arithmetic.fence.f32(float %mul)
  %add = fadd fast float %tmp, %c
  ret float %add
}

define double @f2(double %a) {
  %1 = fadd fast double %a, %a
  %t = call double @llvm.arithmetic.fence.f64(double %1)
  %2 = fadd fast double %a, %a
  %3 = fadd fast double %t, %2
  ret double %3
}

define <2 x float> @f3(<2 x float> %a) {
  %1 = fadd fast <2 x float> %a, %a
  %t = call <2 x float> @llvm.arithmetic.fence.v2f32(<2 x float> %1)
  %2 = fadd fast <2 x float> %a, %a
  %3 = fadd fast <2 x float> %t, %2
  ret <2 x float> %3
}

declare float @llvm.arithmetic.fence.f32(float)
declare double @llvm.arithmetic.fence.f64(double)
declare <2 x float> @llvm.arithmetic.fence.v2f32(<2 x float>)
