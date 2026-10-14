; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=FPFastMathMode
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Double2:]] = OpTypeVector %[[#Double]] 2
; CHECK-DAG: %[[#Zero:]] = OpConstantNull %[[#Float]]
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Float]] 1

; CHECK: %[[#X:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Y:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#XNaN:]] = OpUnordered %[[#]] %[[#X]] %[[#X]]
; CHECK: %[[#X1:]] = OpSelect %[[#Float]] %[[#XNaN]] %[[#Y]] %[[#X]]
; CHECK: %[[#YNaN:]] = OpUnordered %[[#]] %[[#Y]] %[[#Y]]
; CHECK: %[[#Y1:]] = OpSelect %[[#Float]] %[[#YNaN]] %[[#X1]] %[[#Y]]
; CHECK: %[[#Min:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmin %[[#X1]] %[[#Y1]]
; CHECK: %[[#XSign:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] copysign %[[#One]] %[[#X1]]
; CHECK: %[[#XNeg:]] = OpFOrdLessThan %[[#]] %[[#XSign]] %[[#Zero]]
; CHECK: %[[#S:]] = OpSelect %[[#Float]] %[[#XNeg]] %[[#X1]] %[[#Y1]]
; CHECK: %[[#R:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] copysign %[[#Min]] %[[#S]]
; CHECK: OpReturnValue %[[#R]]
define spir_func float @test_minimumnum(float %x, float %y) {
  %r = call float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

; CHECK: OpFunctionParameter %[[#Double2]]
; CHECK: %[[#Max:]] = OpExtInst %[[#Double2]] %[[#ExtInstId]] fmax %[[#Xv:]] %[[#Yv:]]
; CHECK: %[[#XNegv:]] = OpFOrdLessThan
; CHECK: %[[#Sv:]] = OpSelect %[[#Double2]] %[[#XNegv]] %[[#Yv]] %[[#Xv]]
; CHECK: %[[#Rv:]] = OpExtInst %[[#Double2]] %[[#ExtInstId]] copysign %[[#Max]] %[[#Sv]]
; CHECK: OpReturnValue %[[#Rv]]
define spir_func <2 x double> @test_maximumnum_v2(<2 x double> %x, <2 x double> %y) {
  %r = call <2 x double> @llvm.maximumnum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %r
}

; CHECK: %[[#Xn:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Yn:]] = OpFunctionParameter %[[#Float]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#Rn:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmin %[[#Xn]] %[[#Yn]]
; CHECK-NEXT: OpReturnValue %[[#Rn]]
define spir_func float @test_minimumnum_nnan_nsz_reassoc(float %x, float %y) {
  %r = call nnan nsz reassoc float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

;; A signaling NaN operand yields the other operand.
; CHECK: OpFunction %[[#Float]] None
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturnValue %[[#One]]
define spir_func float @test_minimumnum_snan() {
  %r = call nsz float @llvm.minimumnum.f32(float 0x7FF0000020000000, float 1.0)
  ret float %r
}
