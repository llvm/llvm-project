; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s

; This test case has two scops in a row. When code generating the first scop,
; the second scop is invalidated. This test case verifies that we do not crash
; due to incorrectly assuming the second scop is still valid.

; We explicitly check here that the second scop is not code generated. Later
; improvements may make this possible (e.g., Polly gaining support for
; parameteric conditional expressions or a changed code generation order).
; However, in case this happens, we want to ensure this test case is been
; reasoned about and updated accordingly.

; CHECK: polly.start:
; CHECK-NOT: polly.start:

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @hoge(i8* %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp = getelementptr inbounds i8, i8* %arg, i64 5
  %tmp2 = getelementptr inbounds i8, i8* %arg, i64 6
  br i1 false, label %bb3, label %bb4

bb3:                                              ; preds = %bb1
  br label %bb4

bb4:                                              ; preds = %bb3, %bb1
  %tmp5 = icmp eq i32 0, 1
  br label %bb6

bb6:                                              ; preds = %bb4
  br i1 undef, label %bb7, label %bb8

bb7:                                              ; preds = %bb6
  unreachable

bb8:                                              ; preds = %bb6
  br i1 %tmp5, label %bb9, label %bb10

bb9:                                              ; preds = %bb8
  br label %bb11

bb10:                                             ; preds = %bb8
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  ret void
}
