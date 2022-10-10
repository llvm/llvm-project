; RUN: opt < %s -aa-pipeline=basic-aa -dse -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.S0 = type <{ i8, [4 x i8] }>

@a = global { i8, i8, i8, i8, i8 } { i8 undef, i8 0, i8 0, i8 0, i8 0 }, align 8

define i32 @main() nounwind uwtable ssp {
entry:
  %tmp = load i8, i8* getelementptr inbounds ({ i8, i8, i8, i8, i8 }, { i8, i8, i8, i8, i8 }* @a, i64 0, i32 4), align 4
  %tmp1 = or i8 %tmp, -128
  store i8 %tmp1, i8* getelementptr inbounds ({ i8, i8, i8, i8, i8 }, { i8, i8, i8, i8, i8 }* @a, i64 0, i32 4), align 4
  %tmp2 = load i64, i64* bitcast ({ i8, i8, i8, i8, i8 }* @a to i64*), align 8
  store i8 11, i8* getelementptr inbounds ({ i8, i8, i8, i8, i8 }, { i8, i8, i8, i8, i8 }* @a, i64 0, i32 4), align 4
  %tmp3 = trunc i64 %tmp2 to i32
  ret i32 %tmp3

; Make sure we don't delete either store here
; CHECK: @main
; CHECK: store i8 %tmp1
; CHECK: store i8 11
}

