; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; PR9011

@pr9011_3 = constant <4 x i32> bitcast (<16 x i8> zeroinitializer to <4 x i32>)
; CHECK: pr9011_3 = constant <4 x i32> zeroinitializer

@pr9011_10 = constant <4 x double> bitcast (i256 0 to <4 x double>)
; CHECK: pr9011_10 = constant <4 x double> zeroinitializer
@pr9011_11 = constant <4 x float> bitcast (i128 0 to <4 x float>)
; CHECK: pr9011_11 = constant <4 x float> zeroinitializer
@pr9011_12 = constant <4 x i32> bitcast (i128 0 to <4 x i32>)
; CHECK: pr9011_12 = constant <4 x i32> zeroinitializer
@pr9011_13 = constant i256 bitcast (<4 x double> zeroinitializer to i256)
; CHECK: pr9011_13 = constant i256 0
@pr9011_14 = constant i128 bitcast (<4 x float> zeroinitializer to i128)
; CHECK: pr9011_14 = constant i128 0
@pr9011_15 = constant i128 bitcast (<4 x i32> zeroinitializer to i128)
; CHECK: pr9011_15 = constant i128 0
