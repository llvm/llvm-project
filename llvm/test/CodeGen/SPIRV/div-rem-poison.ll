; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown < %s -filetype=obj | spirv-val %}

; OpSDiv, OpUDiv, OpSRem and OpUMod are undefined for a zero divisor (and the
; signed ones for the minimum value divided by -1), so PreISelIntrinsicLowering
; expands every call into a plain instruction with a guarded divisor: the
; OpSelect that replaces the divisor has to precede the division. The compare
; and the select are element-wise for a vector.

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Int]] 1
; CHECK-DAG: %[[#MinusOne:]] = OpConstant %[[#Int]] 4294967295
; CHECK-DAG: %[[#SignedMin:]] = OpConstant %[[#Int]] 2147483648
; CHECK-DAG: %[[#Null:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#VecInt:]] = OpTypeVector %[[#Int]] 4
; CHECK-DAG: %[[#VecBool:]] = OpTypeVector %[[#Bool]] 4
; CHECK-DAG: %[[#VecSignedMin:]] = OpConstantComposite %[[#VecInt]] %[[#SignedMin]] %[[#SignedMin]] %[[#SignedMin]] %[[#SignedMin]]
; CHECK-DAG: %[[#VecMinusOne:]] = OpConstantComposite %[[#VecInt]] %[[#MinusOne]] %[[#MinusOne]] %[[#MinusOne]] %[[#MinusOne]]
; CHECK-DAG: %[[#VecOne:]] = OpConstantComposite %[[#VecInt]] %[[#One]] %[[#One]] %[[#One]] %[[#One]]
; CHECK-DAG: %[[#VecNull:]] = OpConstantNull %[[#VecInt]]

define i32 @sdiv_i32(i32 %a, i32 %b) {
; CHECK: ; -- Begin function sdiv_i32
; CHECK-NEXT: %[[#Dividend:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#Divisor:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#]] = OpLabel
; CHECK-NEXT: %[[#IsZero:]] = OpIEqual %[[#Bool]] %[[#Divisor]] %[[#Null]]
; CHECK-NEXT: %[[#IsMin:]] = OpIEqual %[[#Bool]] %[[#Dividend]] %[[#SignedMin]]
; CHECK-NEXT: %[[#IsMinusOne:]] = OpIEqual %[[#Bool]] %[[#Divisor]] %[[#MinusOne]]
; CHECK-NEXT: %[[#IsOverflow:]] = OpLogicalAnd %[[#Bool]] %[[#IsMin]] %[[#IsMinusOne]]
; CHECK-NEXT: %[[#IsBad:]] = OpLogicalOr %[[#Bool]] %[[#IsZero]] %[[#IsOverflow]]
; CHECK-NEXT: %[[#SafeDivisor:]] = OpSelect %[[#Int]] %[[#IsBad]] %[[#One]] %[[#Divisor]]
; CHECK-NEXT: %[[#Quotient:]] = OpSDiv %[[#Int]] %[[#Dividend]] %[[#SafeDivisor]]
; CHECK-NEXT: OpReturnValue %[[#Quotient]]
; CHECK-NEXT: OpFunctionEnd
  %r = call i32 @llvm.sdiv.poison.i32(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @udiv_i32(i32 %a, i32 %b) {
; CHECK: ; -- Begin function udiv_i32
; CHECK-NEXT: %[[#Dividend:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#Divisor:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#]] = OpLabel
; CHECK-NEXT: %[[#IsZero:]] = OpIEqual %[[#Bool]] %[[#Divisor]] %[[#Null]]
; CHECK-NEXT: %[[#SafeDivisor:]] = OpSelect %[[#Int]] %[[#IsZero]] %[[#One]] %[[#Divisor]]
; CHECK-NEXT: %[[#Quotient:]] = OpUDiv %[[#Int]] %[[#Dividend]] %[[#SafeDivisor]]
; CHECK-NEXT: OpReturnValue %[[#Quotient]]
; CHECK-NEXT: OpFunctionEnd
  %r = call i32 @llvm.udiv.poison.i32(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @srem_i32(i32 %a, i32 %b) {
; CHECK: ; -- Begin function srem_i32
; CHECK-NEXT: %[[#Dividend:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#Divisor:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#]] = OpLabel
; CHECK-NEXT: %[[#IsZero:]] = OpIEqual %[[#Bool]] %[[#Divisor]] %[[#Null]]
; CHECK-NEXT: %[[#IsMin:]] = OpIEqual %[[#Bool]] %[[#Dividend]] %[[#SignedMin]]
; CHECK-NEXT: %[[#IsMinusOne:]] = OpIEqual %[[#Bool]] %[[#Divisor]] %[[#MinusOne]]
; CHECK-NEXT: %[[#IsOverflow:]] = OpLogicalAnd %[[#Bool]] %[[#IsMin]] %[[#IsMinusOne]]
; CHECK-NEXT: %[[#IsBad:]] = OpLogicalOr %[[#Bool]] %[[#IsZero]] %[[#IsOverflow]]
; CHECK-NEXT: %[[#SafeDivisor:]] = OpSelect %[[#Int]] %[[#IsBad]] %[[#One]] %[[#Divisor]]
; CHECK-NEXT: %[[#Remainder:]] = OpSRem %[[#Int]] %[[#Dividend]] %[[#SafeDivisor]]
; CHECK-NEXT: OpReturnValue %[[#Remainder]]
; CHECK-NEXT: OpFunctionEnd
  %r = call i32 @llvm.srem.poison.i32(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @urem_i32(i32 %a, i32 %b) {
; CHECK: ; -- Begin function urem_i32
; CHECK-NEXT: %[[#Dividend:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#Divisor:]] = OpFunctionParameter %[[#Int]]
; CHECK-NEXT: %[[#]] = OpLabel
; CHECK-NEXT: %[[#IsZero:]] = OpIEqual %[[#Bool]] %[[#Divisor]] %[[#Null]]
; CHECK-NEXT: %[[#SafeDivisor:]] = OpSelect %[[#Int]] %[[#IsZero]] %[[#One]] %[[#Divisor]]
; CHECK-NEXT: %[[#Remainder:]] = OpUMod %[[#Int]] %[[#Dividend]] %[[#SafeDivisor]]
; CHECK-NEXT: OpReturnValue %[[#Remainder]]
; CHECK-NEXT: OpFunctionEnd
  %r = call i32 @llvm.urem.poison.i32(i32 %a, i32 %b)
  ret i32 %r
}

define <4 x i32> @sdiv_v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: ; -- Begin function sdiv_v4i32
; CHECK-NEXT: %[[#Dividend:]] = OpFunctionParameter %[[#VecInt]]
; CHECK-NEXT: %[[#Divisor:]] = OpFunctionParameter %[[#VecInt]]
; CHECK-NEXT: %[[#]] = OpLabel
; CHECK-NEXT: %[[#IsZero:]] = OpIEqual %[[#VecBool]] %[[#Divisor]] %[[#VecNull]]
; CHECK-NEXT: %[[#IsMin:]] = OpIEqual %[[#VecBool]] %[[#Dividend]] %[[#VecSignedMin]]
; CHECK-NEXT: %[[#IsMinusOne:]] = OpIEqual %[[#VecBool]] %[[#Divisor]] %[[#VecMinusOne]]
; CHECK-NEXT: %[[#IsOverflow:]] = OpLogicalAnd %[[#VecBool]] %[[#IsMin]] %[[#IsMinusOne]]
; CHECK-NEXT: %[[#IsBad:]] = OpLogicalOr %[[#VecBool]] %[[#IsZero]] %[[#IsOverflow]]
; CHECK-NEXT: %[[#SafeDivisor:]] = OpSelect %[[#VecInt]] %[[#IsBad]] %[[#VecOne]] %[[#Divisor]]
; CHECK-NEXT: %[[#Quotient:]] = OpSDiv %[[#VecInt]] %[[#Dividend]] %[[#SafeDivisor]]
; CHECK-NEXT: OpReturnValue %[[#Quotient]]
; CHECK-NEXT: OpFunctionEnd
  %r = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

declare i32 @llvm.sdiv.poison.i32(i32, i32)
declare i32 @llvm.udiv.poison.i32(i32, i32)
declare i32 @llvm.srem.poison.i32(i32, i32)
declare i32 @llvm.urem.poison.i32(i32, i32)
declare <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32>, <4 x i32>)
