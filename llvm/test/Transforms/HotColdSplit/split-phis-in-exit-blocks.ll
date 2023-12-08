; RUN: opt -hotcoldsplit-threshold=0 -passes=hotcoldsplit -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@pluto(
; CHECK-NEXT: bb:
; CHECK-NEXT:  %tmp8.ce.loc = alloca i1
; CHECK-NEXT:  switch i8 undef, label %codeRepl [
; CHECK-NEXT:    i8 0, label %bb7
; CHECK-NEXT:    i8 1, label %bb7
; CHECK-NEXT:  ]
;
; CHECK:  codeRepl:
; CHECK-NEXT:    lifetime.start
; CHECK-NEXT:    call void @pluto.cold.1(ptr %tmp8.ce.loc)
; CHECK-NEXT:    %tmp8.ce.reload = load i1, ptr %tmp8.ce.loc
; CHECK-NEXT:    lifetime.end
; CHECK-NEXT:    br label %bb7
;
; CHECK:  bb7:
; CHECK:    %tmp8 = phi i1 [ true, %bb ], [ true, %bb ], [ %tmp8.ce.reload, %codeRepl ]
; CHECK:    ret void

; CHECK-LABEL: define {{.*}}@pluto.cold.1(
; CHECK: call {{.*}}@sideeffect(i32 1)
; CHECK: call {{.*}}@sink(
; CHECK: call {{.*}}@sideeffect(i32 3)
; CHECK: call {{.*}}@sideeffect(i32 4)
; CHECK: call {{.*}}@sideeffect(i32 5)
define void @pluto() {
bb:
  switch i8 undef, label %bb1 [
    i8 0, label %bb7
    i8 1, label %bb7
  ]

bb1:                                              ; preds = %bb
  call void @sideeffect(i32 1)
  br label %bb2

bb2:                                              ; preds = %bb1
  call void @sink()
  br i1 undef, label %bb7, label %bb3

bb3:                                              ; preds = %bb2
  call void @sideeffect(i32 3)
  br label %bb4

bb4:                                              ; preds = %bb3
  call void @sideeffect(i32 4)
  br i1 undef, label %bb5, label %bb6

bb5:                                              ; preds = %bb4
  call void @sideeffect(i32 5)
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4
  %tmp = phi i1 [ true, %bb5 ], [ false, %bb4 ]
  call void @sideeffect(i32 6)
  br label %bb7

bb7:                                              ; preds = %bb6, %bb2, %bb, %bb
  %tmp8 = phi i1 [ true, %bb ], [ true, %bb ], [ true, %bb2 ], [ %tmp, %bb6 ]
  ret void
}

declare void @sink() cold

declare void @sideeffect(i32)
