; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

; Function Attrs: uwtable
define void @test1() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw(i32 3)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  tail call void @may_throw(i32 2)
  tail call void @may_throw(i32 1)
  ret void

ehcleanup:                                        ; preds = %entry
  %cp = cleanuppad within none []
  tail call void @may_throw(i32 2) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind label %ehcleanup2

ehcleanup2:
  %cp2 = cleanuppad within none []
  tail call void @may_throw(i32 1) [ "funclet"(token %cp2) ]
  cleanupret from %cp2 unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: %[[cp:.*]] = cleanuppad within none []
; CHECK: tail call void @may_throw(i32 2) [ "funclet"(token %[[cp]]) ]
; CHECK: tail call void @may_throw(i32 1) [ "funclet"(token %[[cp]]) ]
; CHECK: cleanupret from %[[cp]] unwind to caller

declare void @may_throw(i32)

declare i32 @__CxxFrameHandler3(...)
