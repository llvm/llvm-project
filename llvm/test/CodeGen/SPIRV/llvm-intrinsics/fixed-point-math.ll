; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Scale16:]] = OpConstant %[[#I16]] 3
; CHECK-DAG: %[[#Scale32:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#Scale64:]] = OpConstant %[[#I64]] 3
; CHECK-DAG: %[[#SMin16:]] = OpConstant %[[#I16]] 65408
; CHECK-DAG: %[[#SMax16:]] = OpConstant %[[#I16]] 127
; CHECK-DAG: %[[#UMax16:]] = OpConstant %[[#I16]] 255
; CHECK-DAG: %[[#SMin32:]] = OpConstant %[[#I32]] 4294934528
; CHECK-DAG: %[[#SMax32:]] = OpConstant %[[#I32]] 32767
; CHECK-DAG: %[[#UMax32:]] = OpConstant %[[#I32]] 65535
; CHECK-DAG: %[[#SMin64:]] = OpConstant %[[#I64]] 18446744071562067968
; CHECK-DAG: %[[#SMax64:]] = OpConstant %[[#I64]] 2147483647
; CHECK-DAG: %[[#UMax64:]] = OpConstant %[[#I64]] 4294967295
; CHECK-DAG: %[[#V4I32:]] = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#V4I64:]] = OpTypeVector %[[#I64]] 4
; CHECK-DAG: %[[#VScale64:]] = OpConstantComposite %[[#V4I64]] %[[#Scale64]] %[[#Scale64]] %[[#Scale64]] %[[#Scale64]]
; CHECK-DAG: %[[#VSMin64:]] = OpConstantComposite %[[#V4I64]] %[[#SMin64]] %[[#SMin64]] %[[#SMin64]] %[[#SMin64]]
; CHECK-DAG: %[[#VSMax64:]] = OpConstantComposite %[[#V4I64]] %[[#SMax64]] %[[#SMax64]] %[[#SMax64]] %[[#SMax64]]
; CHECK-DAG: %[[#VUMax64:]] = OpConstantComposite %[[#V4I64]] %[[#UMax64]] %[[#UMax64]] %[[#UMax64]] %[[#UMax64]]

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I16]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I16]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I16]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I16]] %[[#Mul]] %[[#Scale16]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I8]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i8 @smulfix_i8(i8 %a, i8 %b) {
  %r = call i8 @llvm.smul.fix.i8(i8 %a, i8 %b, i32 3)
  ret i8 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I16]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I16]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I16]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I16]] %[[#Mul]] %[[#Scale16]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I8]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i8 @umulfix_i8(i8 %a, i8 %b) {
  %r = call i8 @llvm.umul.fix.i8(i8 %a, i8 %b, i32 3)
  ret i8 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I32]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I32]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I32]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I32]] %[[#Mul]] %[[#Scale32]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I16]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i16 @smulfix_i16(i16 %a, i16 %b) {
  %r = call i16 @llvm.smul.fix.i16(i16 %a, i16 %b, i32 3)
  ret i16 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I32]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I32]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I32]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I32]] %[[#Mul]] %[[#Scale32]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I16]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i16 @umulfix_i16(i16 %a, i16 %b) {
  %r = call i16 @llvm.umul.fix.i16(i16 %a, i16 %b, i32 3)
  ret i16 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I64]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I64]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I64]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I64]] %[[#Mul]] %[[#Scale64]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I32]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i32 @smulfix_i32(i32 %a, i32 %b) {
  %r = call i32 @llvm.smul.fix.i32(i32 %a, i32 %b, i32 3)
  ret i32 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I64]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I64]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I64]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I64]] %[[#Mul]] %[[#Scale64]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I32]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i32 @umulfix_i32(i32 %a, i32 %b) {
  %r = call i32 @llvm.umul.fix.i32(i32 %a, i32 %b, i32 3)
  ret i32 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I16]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I16]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I16]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I16]] %[[#Mul]] %[[#Scale16]]
; CHECK: %[[#Min:]] = OpExtInst %[[#I16]] %[[#]] s_min %[[#Shift]] %[[#SMax16]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#I16]] %[[#]] s_max %[[#Min]] %[[#SMin16]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I8]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define i8 @smulfixsat_i8(i8 %a, i8 %b) {
  %r = call i8 @llvm.smul.fix.sat.i8(i8 %a, i8 %b, i32 3)
  ret i8 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I16]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I16]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I16]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I16]] %[[#Mul]] %[[#Scale16]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#I16]] %[[#]] u_min %[[#Shift]] %[[#UMax16]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I8]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define i8 @umulfixsat_i8(i8 %a, i8 %b) {
  %r = call i8 @llvm.umul.fix.sat.i8(i8 %a, i8 %b, i32 3)
  ret i8 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I32]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I32]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I32]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I32]] %[[#Mul]] %[[#Scale32]]
; CHECK: %[[#Min:]] = OpExtInst %[[#I32]] %[[#]] s_min %[[#Shift]] %[[#SMax32]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#I32]] %[[#]] s_max %[[#Min]] %[[#SMin32]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I16]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define i16 @smulfixsat_i16(i16 %a, i16 %b) {
  %r = call i16 @llvm.smul.fix.sat.i16(i16 %a, i16 %b, i32 3)
  ret i16 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I32]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I32]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I32]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I32]] %[[#Mul]] %[[#Scale32]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#I32]] %[[#]] u_min %[[#Shift]] %[[#UMax32]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I16]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define i16 @umulfixsat_i16(i16 %a, i16 %b) {
  %r = call i16 @llvm.umul.fix.sat.i16(i16 %a, i16 %b, i32 3)
  ret i16 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I64]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I64]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I64]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I64]] %[[#Mul]] %[[#Scale64]]
; CHECK: %[[#Min:]] = OpExtInst %[[#I64]] %[[#]] s_min %[[#Shift]] %[[#SMax64]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#I64]] %[[#]] s_max %[[#Min]] %[[#SMin64]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I32]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define i32 @smulfixsat_i32(i32 %a, i32 %b) {
  %r = call i32 @llvm.smul.fix.sat.i32(i32 %a, i32 %b, i32 3)
  ret i32 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I64]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I64]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I64]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I64]] %[[#Mul]] %[[#Scale64]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#I64]] %[[#]] u_min %[[#Shift]] %[[#UMax64]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I32]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define i32 @umulfixsat_i32(i32 %a, i32 %b) {
  %r = call i32 @llvm.umul.fix.sat.i32(i32 %a, i32 %b, i32 3)
  ret i32 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#V4I64]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#V4I64]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#V4I64]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#V4I64]] %[[#Mul]] %[[#VScale64]]
; CHECK: %[[#Min:]] = OpExtInst %[[#V4I64]] %[[#]] s_min %[[#Shift]] %[[#VSMax64]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#V4I64]] %[[#]] s_max %[[#Min]] %[[#VSMin64]]
; CHECK: %[[#Res:]] = OpUConvert %[[#V4I32]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define <4 x i32> @smulfixsat_v4i32(<4 x i32> %a, <4 x i32> %b) {
  %r = call <4 x i32> @llvm.smul.fix.sat.v4i32(<4 x i32> %a, <4 x i32> %b, i32 3)
  ret <4 x i32> %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#V4I64]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#V4I64]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#V4I64]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#V4I64]] %[[#Mul]] %[[#VScale64]]
; CHECK: %[[#Clamp:]] = OpExtInst %[[#V4I64]] %[[#]] u_min %[[#Shift]] %[[#VUMax64]]
; CHECK: %[[#Res:]] = OpUConvert %[[#V4I32]] %[[#Clamp]]
; CHECK: OpReturnValue %[[#Res]]
define <4 x i32> @umulfixsat_v4i32(<4 x i32> %a, <4 x i32> %b) {
  %r = call <4 x i32> @llvm.umul.fix.sat.v4i32(<4 x i32> %a, <4 x i32> %b, i32 3)
  ret <4 x i32> %r
}
