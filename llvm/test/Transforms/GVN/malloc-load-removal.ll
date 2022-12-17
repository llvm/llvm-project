; RUN: opt -S -passes=gvn < %s | FileCheck %s
; PR13694

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare noalias i8* @malloc(i64) nounwind allockind("alloc,uninitialized") allocsize(0)

define noalias i8* @test1() nounwind uwtable ssp {
entry:
  %call = tail call i8* @malloc(i64 100) nounwind
  %0 = load i8, i8* %call, align 1
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i8 0, i8* %call, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i8* %call

; CHECK-LABEL: @test1(
; CHECK-NOT: load
; CHECK-NOT: icmp

; CHECK_NO_LIBCALLS-LABEL: @test1(
; CHECK_NO_LIBCALLS: load
; CHECK_NO_LIBCALLS: icmp
}

declare noalias i8* @_Znwm(i64) nounwind

define noalias i8* @test2() nounwind uwtable ssp {
entry:
  %call = tail call i8* @_Znwm(i64 100) nounwind
  %0 = load i8, i8* %call, align 1
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i8 0, i8* %call, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i8* %call

; CHECK-LABEL: @test2(
; CHECK-NOT: load
; CHECK-NOT: icmp

; CHECK_NO_LIBCALLS-LABEL: @test2(
; CHECK_NO_LIBCALLS: load
; CHECK_NO_LIBCALLS: icmp
}

declare noalias i8* @aligned_alloc(i64 allocalign, i64) nounwind allockind("alloc,uninitialized,aligned") allocsize(1)

define noalias i8* @test3() nounwind uwtable ssp {
entry:
  %call = tail call i8* @aligned_alloc(i64 256, i64 32) nounwind
  %0 = load i8, i8* %call, align 32
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i8 0, i8* %call, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i8* %call

; CHECK-LABEL: @test3(
; CHECK-NOT: load
; CHECK-NOT: icmp

; CHECK_NO_LIBCALLS-LABEL: @test3(
; CHECK_NO_LIBCALLS: load
; CHECK_NO_LIBCALLS: icmp
}
