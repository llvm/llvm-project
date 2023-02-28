; RUN: opt -S -passes=guard-widening,verify < %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; REQUIRES: asserts
; XFAIL: *

; GuardWidening moves 'check' instruction before 'wc1' without its operand ('zero').
; This is incorrect as 'zero' is not available at that point.
define void @foo() {
bb:
  %wc1 = call i1 @llvm.experimental.widenable.condition()
  %wc2 = call i1 @llvm.experimental.widenable.condition()
  %zero = add i32 0, 0
  %check = icmp ult i32 %zero, 0
  %c1 = and i1 %check, %wc1
  %c2 = and i1 %check, %wc2
  br i1 %c1, label %bb6, label %bb9

bb6:                                              ; preds = %bb
  br i1 %c2, label %bb7, label %bb8

bb7:                                              ; preds = %bb6
  ret void

bb8:                                              ; preds = %bb6
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 0) [ "deopt"() ]
  ret void

bb9:                                              ; preds = %bb
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 0) [ "deopt"() ]
  ret void
}

declare void @llvm.experimental.deoptimize.isVoid(...)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite)
declare i1 @llvm.experimental.widenable.condition() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite) }
