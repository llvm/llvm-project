; RUN: opt -S -passes=hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx27.0.0"

declare void @sink_call(ptr, i32, ptr)

; CHECK-LABEL: define ptr @test_stale_funcretval(
; CHECK: call void @test_stale_funcretval.cold.1()
; CHECK-LABEL: end:
; CHECK-NEXT: phi ptr [ %ph, %codeRepl ], [ null, %then ]

; CHECK-LABEL: define internal void @test_stale_funcretval.cold.1()
; CHECK: ret void
define ptr @test_stale_funcretval() {
entry:
  br i1 false, label %sink, label %then

then:                                             ; preds = %entry
  br i1 false, label %end, label %sink, !prof !0

sink:                                             ; preds = %then, %entry
  %ph = phi ptr [ null, %then ], [ null, %entry ]
  tail call void @sink_call(ptr null, i32 0, ptr null)
  br label %end

end:                                              ; preds = %sink, %then
  %val = phi ptr [ %ph, %sink ], [ null, %then ]
  ret ptr %val
}

!0 = !{!"branch_weights", !"expected", i32 2000, i32 1}
