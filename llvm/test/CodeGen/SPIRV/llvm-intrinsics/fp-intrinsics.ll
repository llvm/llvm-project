; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"

; CHECK: %[[#var0:]] = OpTypeFloat 16
; CHECK: %[[#var1:]] = OpTypeFloat 32
; CHECK: %[[#var2:]] = OpTypeFloat 64
; CHECK: %[[#var3:]] = OpTypeVector %[[#var1]] 4

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var0]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func half @TestFabs16(half %x) local_unnamed_addr {
entry:
  %t = tail call half @llvm.fabs.f16(half %x)
  ret half %t
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func float @TestFabs32(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.fabs.f32(float %x)
  ret float %t
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var2]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func double @TestFabs64(double %x) local_unnamed_addr {
entry:
  %t = tail call double @llvm.fabs.f64(double %x)
  ret double %t
}

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var3]] %[[#extinst_id]] fabs
; CHECK: OpFunctionEnd

define spir_func <4 x float> @TestFabsVec(<4 x float> %x) local_unnamed_addr {
entry:
  %t = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  ret <4 x float> %t
}

declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

;; We checked several types with fabs, but the type check works the same for
;; all intrinsics being translated, so for the rest we'll just test one type.

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] ceil
; CHECK: OpFunctionEnd

define spir_func float @TestCeil(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.ceil.f32(float %x)
  ret float %t
}

declare float @llvm.ceil.f32(float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#n:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] pown %[[#x]] %[[#n]]
; CHECK: OpFunctionEnd

define spir_func float @TestPowi(float %x, i32 %n) local_unnamed_addr {
entry:
  %t = tail call float @llvm.powi.f32(float %x, i32 %n)
  ret float %t
}

declare float @llvm.powi.f32(float, i32)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] sin
; CHECK: OpFunctionEnd

define spir_func float @TestSin(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.sin.f32(float %x)
  ret float %t
}

declare float @llvm.sin.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] cos
; CHECK: OpFunctionEnd

define spir_func float @TestCos(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.cos.f32(float %x)
  ret float %t
}

declare float @llvm.cos.f32(float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] pow %[[#x]] %[[#y]]
; CHECK: OpFunctionEnd

define spir_func float @TestPow(float %x, float %y) local_unnamed_addr {
entry:
  %t = tail call float @llvm.pow.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.pow.f32(float, float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] exp
; CHECK: OpFunctionEnd

define spir_func float @TestExp(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.exp.f32(float %x)
  ret float %t
}

declare float @llvm.exp.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] exp2
; CHECK: OpFunctionEnd

define spir_func float @TestExp2(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.exp2.f32(float %x)
  ret float %t
}

declare float @llvm.exp2.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] log
; CHECK: OpFunctionEnd

define spir_func float @TestLog(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.log.f32(float %x)
  ret float %t
}

declare float @llvm.log.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] log10
; CHECK: OpFunctionEnd

define spir_func float @TestLog10(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.log10.f32(float %x)
  ret float %t
}

declare float @llvm.log10.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] log2
; CHECK: OpFunctionEnd

define spir_func float @TestLog2(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.log2.f32(float %x)
  ret float %t
}

declare float @llvm.log2.f32(float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#res:]] = OpExtInst %[[#]] %[[#]] fmin %[[#x]] %[[#y]]
; CHECK: OpReturnValue %[[#res]]

define spir_func float @TestMinNum(float %x, float %y) {
entry:
  %t = call float @llvm.minnum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.minnum.f32(float, float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#res:]] = OpExtInst %[[#]] %[[#]] fmax %[[#x]] %[[#y]]
; CHECK: OpReturnValue %[[#res]]

define spir_func float @TestMaxNum(float %x, float %y) {
entry:
  %t = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.maxnum.f32(float, float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#res:]] = OpExtInst %[[#]] %[[#]] fmin %[[#x]] %[[#y]]
; CHECK: OpReturnValue %[[#res]]

define spir_func float @TestMinimum(float %x, float %y) {
entry:
  %t = call float @llvm.minimum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.minimum.f32(float, float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#res:]] = OpExtInst %[[#]] %[[#]] fmax %[[#x]] %[[#y]]
; CHECK: OpReturnValue %[[#res]]

define spir_func float @TestMaximum(float %x, float %y) {
entry:
  %t = call float @llvm.maximum.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.maximum.f32(float, float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] copysign %[[#x]] %[[#y]]
; CHECK: OpFunctionEnd

define spir_func float @TestCopysign(float %x, float %y) local_unnamed_addr {
entry:
  %t = tail call float @llvm.copysign.f32(float %x, float %y)
  ret float %t
}

declare float @llvm.copysign.f32(float, float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] floor
; CHECK: OpFunctionEnd

define spir_func float @TestFloor(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.floor.f32(float %x)
  ret float %t
}

declare float @llvm.floor.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] trunc
; CHECK: OpFunctionEnd

define spir_func float @TestTrunc(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.trunc.f32(float %x)
  ret float %t
}

declare float @llvm.trunc.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] rint
; CHECK: OpFunctionEnd

define spir_func float @TestRint(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.rint.f32(float %x)
  ret float %t
}

declare float @llvm.rint.f32(float)

;; It is intentional that nearbyint translates to rint.
; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] rint
; CHECK: OpFunctionEnd

define spir_func float @TestNearbyint(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.nearbyint.f32(float %x)
  ret float %t
}

declare float @llvm.nearbyint.f32(float)

; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] round
; CHECK: OpFunctionEnd

define spir_func float @TestRound(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.round.f32(float %x)
  ret float %t
}

declare float @llvm.round.f32(float)

;; It is intentional that roundeven translates to rint.
; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] rint
; CHECK: OpFunctionEnd

define spir_func float @TestRoundEven(float %x) local_unnamed_addr {
entry:
  %t = tail call float @llvm.roundeven.f32(float %x)
  ret float %t
}

declare float @llvm.roundeven.f32(float)

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#z:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#var1]] %[[#extinst_id]] fma %[[#x]] %[[#y]] %[[#z]]
; CHECK: OpFunctionEnd

define spir_func float @TestFma(float %x, float %y, float %z) {
entry:
  %t = tail call float @llvm.fma.f32(float %x, float %y, float %z)
  ret float %t
}

declare float @llvm.fma.f32(float, float, float)

; CHECK: OpFunction
; CHECK: %[[#d:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#fracPtr:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#integralPtr:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#varPtr:]] = OpVariable %[[#]] Function
; CHECK: %[[#frac:]] = OpExtInst %[[#var2]] %[[#extinst_id]] modf %[[#d]] %[[#varPtr]]
; CHECK: %[[#integral:]] = OpLoad %[[#var2]] %[[#varPtr]]
; CHECK: OpStore %[[#fracPtr]] %[[#frac]]
; CHECK: OpStore %[[#integralPtr]] %[[#integral]]
; CHECK: OpFunctionEnd
define void @TestModf(double %d, ptr addrspace(1) %frac, ptr addrspace(1) %integral) {
entry:
  %4 = tail call { double, double } @llvm.modf.f64(double %d)
  %5 = extractvalue { double, double } %4, 0
  %6 = extractvalue { double, double } %4, 1
  store double %5, ptr addrspace(1) %frac, align 8
  store double %6, ptr addrspace(1) %integral, align 8
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#d:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#fracPtr:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#integralPtr:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#entryBlock:]] = OpLabel
; CHECK: %[[#varPtr:]] = OpVariable %[[#]] Function
; CHECK: OpBranchConditional %[[#]] %[[#lor_lhs_falseBlock:]] %[[#if_thenBlock:]]
; CHECK: %[[#lor_lhs_falseBlock]] = OpLabel
; CHECK: OpBranchConditional %[[#]] %[[#if_endBlock:]] %[[#if_thenBlock]]
; CHECK: %[[#if_thenBlock]] = OpLabel
; CHECK: OpBranch %[[#returnBlock:]]
; CHECK: %[[#if_endBlock]] = OpLabel
; CHECK: %[[#frac:]] = OpExtInst %[[#var2]] %[[#extinst_id]] modf %[[#d]] %[[#varPtr]]
; CHECK: %[[#integral:]] = OpLoad %[[#var2]] %[[#varPtr]]
; CHECK: OpStore %[[#fracPtr]] %[[#frac]]
; CHECK: OpStore %[[#integralPtr]] %[[#integral]]
; CHECK: OpFunctionEnd
define dso_local void @TestModf2(double noundef %d, ptr noundef %frac, ptr noundef %integral) {
entry:
  %0 = load ptr, ptr %frac, align 8
  %tobool = icmp ne ptr %0, null
  br i1 %tobool, label %lor.lhs.false, label %if.then

lor.lhs.false:
  %1 = load ptr, ptr %integral, align 8
  %tobool1 = icmp ne ptr %1, null
  br i1 %tobool1, label %if.end, label %if.then

if.then:
  br label %return

if.end:
  %6 = tail call { double, double } @llvm.modf.f64(double %d)
  %7 = extractvalue { double, double } %6, 0
  %8 = extractvalue { double, double } %6, 1
  store double %7, ptr %frac, align 4
  store double %8, ptr %integral, align 4
  br label %return

return:
  ret void
}

declare { double, double } @llvm.modf.f64(double)
