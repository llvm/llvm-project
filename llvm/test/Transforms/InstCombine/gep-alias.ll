; RUN: opt -S -instcombine -o - %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android10000"

@x.hwasan = private global { [3 x i32], [4 x i8] } { [3 x i32] [i32 42, i32 57, i32 10], [4 x i8] c"\00\00\00\87" }, align 16
@x = alias [3 x i32], inttoptr (i64 add (i64 ptrtoint ({ [3 x i32], [4 x i8] }* @x.hwasan to i64), i64 -8718968878589280256) to [3 x i32]*)

define i32 @f(i64 %i) {
entry:
  ; CHECK: getelementptr inbounds [3 x i32], [3 x i32]* @x
  %arrayidx = getelementptr inbounds [3 x i32], [3 x i32]* @x, i64 0, i64 %i
  %0 = load i32, i32* %arrayidx
  ret i32 %0
}
