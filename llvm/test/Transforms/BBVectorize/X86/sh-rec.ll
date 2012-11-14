target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -S | FileCheck %s

define void @ptoa() nounwind uwtable {
entry:
  %call = call i8* @malloc() nounwind
  br i1 undef, label %return, label %if.end10

if.end10:                                         ; preds = %entry
  %incdec.ptr = getelementptr inbounds i8* %call, i64 undef
  %call17 = call i32 @ptou() nounwind
  %incdec.ptr26.1 = getelementptr inbounds i8* %incdec.ptr, i64 -2
  store i8 undef, i8* %incdec.ptr26.1, align 1
  %div27.1 = udiv i32 %call17, 100
  %rem.2 = urem i32 %div27.1, 10
  %add2230.2 = or i32 %rem.2, 48
  %conv25.2 = trunc i32 %add2230.2 to i8
  %incdec.ptr26.2 = getelementptr inbounds i8* %incdec.ptr, i64 -3
  store i8 %conv25.2, i8* %incdec.ptr26.2, align 1
  %incdec.ptr26.3 = getelementptr inbounds i8* %incdec.ptr, i64 -4
  store i8 undef, i8* %incdec.ptr26.3, align 1
  %div27.3 = udiv i32 %call17, 10000
  %rem.4 = urem i32 %div27.3, 10
  %add2230.4 = or i32 %rem.4, 48
  %conv25.4 = trunc i32 %add2230.4 to i8
  %incdec.ptr26.4 = getelementptr inbounds i8* %incdec.ptr, i64 -5
  store i8 %conv25.4, i8* %incdec.ptr26.4, align 1
  %div27.4 = udiv i32 %call17, 100000
  %rem.5 = urem i32 %div27.4, 10
  %add2230.5 = or i32 %rem.5, 48
  %conv25.5 = trunc i32 %add2230.5 to i8
  %incdec.ptr26.5 = getelementptr inbounds i8* %incdec.ptr, i64 -6
  store i8 %conv25.5, i8* %incdec.ptr26.5, align 1
  %incdec.ptr26.6 = getelementptr inbounds i8* %incdec.ptr, i64 -7
  store i8 0, i8* %incdec.ptr26.6, align 1
  %incdec.ptr26.7 = getelementptr inbounds i8* %incdec.ptr, i64 -8
  store i8 undef, i8* %incdec.ptr26.7, align 1
  %div27.7 = udiv i32 %call17, 100000000
  %rem.8 = urem i32 %div27.7, 10
  %add2230.8 = or i32 %rem.8, 48
  %conv25.8 = trunc i32 %add2230.8 to i8
  %incdec.ptr26.8 = getelementptr inbounds i8* %incdec.ptr, i64 -9
  store i8 %conv25.8, i8* %incdec.ptr26.8, align 1
  unreachable

return:                                           ; preds = %entry
  ret void
; CHECK: @ptoa
}

declare noalias i8* @malloc() nounwind

declare i32 @ptou()
