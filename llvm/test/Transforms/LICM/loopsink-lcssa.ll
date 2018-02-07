; XFAIL: *
; ...should pass when LoopSink stops breaking LCSSA. Currently the test fails
; when expensive checks are enabled. Also, a LCSSA verification if disabled in
; loop pass manager until LoopSink is fixed.
; RUN: opt -S -loop-sink -verify-loop-lcssa < %s | FileCheck %s
; RUN: opt -S -aa-pipeline=basic-aa -passes=loop-sink -verify-loop-lcssa < %s | FileCheck %s
%a = type { i8 }

; CHECK-LABEL: @foo
; CHECK: ret void
define void @foo() !prof !0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp = getelementptr inbounds %a, %a* undef, i64 0, i32 0
  br label %bb2

bb2:                                              ; preds = %bb16, %bb1
  br i1 undef, label %bb16, label %bb3

bb3:                                              ; preds = %bb2
  br i1 undef, label %bb16, label %bb4

bb4:                                              ; preds = %bb3
  br i1 undef, label %bb5, label %bb16

bb5:                                              ; preds = %bb4
  br i1 undef, label %bb16, label %bb6

bb6:                                              ; preds = %bb5
  br i1 undef, label %bb16, label %bb7

bb7:                                              ; preds = %bb15, %bb6
  br i1 undef, label %bb8, label %bb16

bb8:                                              ; preds = %bb7
  br i1 undef, label %bb9, label %bb15

bb9:                                              ; preds = %bb8
  br i1 undef, label %bb10, label %bb15

bb10:                                             ; preds = %bb9
  br i1 undef, label %bb11, label %bb15

bb11:                                             ; preds = %bb10
  br i1 undef, label %bb12, label %bb15

bb12:                                             ; preds = %bb11
  %tmp13 = load i8, i8* %tmp, align 8
  br i1 undef, label %bb15, label %bb14

bb14:                                             ; preds = %bb12
  call void @bar(i8* %tmp)
  br label %bb16

bb15:                                             ; preds = %bb12, %bb11, %bb10, %bb9, %bb8
  br i1 undef, label %bb16, label %bb7

bb16:                                             ; preds = %bb15, %bb14, %bb7, %bb6, %bb5, %bb4, %bb3, %bb2
  br i1 undef, label %bb17, label %bb2

bb17:                                             ; preds = %bb16
  ret void
}

declare void @bar(i8*)

!0 = !{!"function_entry_count", i64 1}
