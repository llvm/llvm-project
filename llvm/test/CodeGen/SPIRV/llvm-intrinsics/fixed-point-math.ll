; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Scale16:]] = OpConstant %[[#I16]] 3
; CHECK-DAG: %[[#Scale32:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#Scale64:]] = OpConstant %[[#I64]] 3

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
