; RUN: opt < %s -licm -S | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc18.0.0"

define void @test1(i32* %s, i1 %b) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = call i32 @pure_computation()
  br i1 %b, label %try.cont, label %while.body

while.body:                                       ; preds = %while.cond
  invoke void @may_throw()
          to label %while.cond unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %while.body
  %.lcssa1 = phi i32 [ %0, %while.body ]
  %cs = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %cp = catchpad within %cs [i8* null, i32 64, i8* null]
  store i32 %.lcssa1, i32* %s
  catchret from %cp to label %try.cont

try.cont:                                         ; preds = %catch, %while.cond
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK: %[[CALL:.*]] = call i32 @pure_computation()
; CHECK: phi i32 [ %[[CALL]]

define void @test2(i32* %s, i1 %b) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = call i32 @pure_computation()
  br i1 %b, label %try.cont, label %while.body

while.body:                                       ; preds = %while.cond
  invoke void @may_throw()
          to label %while.cond unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %while.body
  %.lcssa1 = phi i32 [ %0, %while.body ]
  %cp = cleanuppad within none []
  store i32 %.lcssa1, i32* %s
  cleanupret from %cp unwind to caller

try.cont:                                         ; preds = %catch, %while.cond
  ret void
}

; CHECK-LABEL: define void @test2(
; CHECK:      %[[CP:.*]] = cleanuppad within none []
; CHECK-NEXT: %[[CALL:.*]] = call i32 @pure_computation() [ "funclet"(token %[[CP]]) ]
; CHECK-NEXT: store i32 %[[CALL]], i32* %s
; CHECK-NEXT: cleanupret from %[[CP]] unwind to caller

declare void @may_throw()

declare i32 @pure_computation() nounwind argmemonly readonly

declare i32 @__CxxFrameHandler3(...)
