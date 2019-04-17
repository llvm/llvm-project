; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @test_load_load_combine_metadata(
; Check that align metadata is combined
; CHECK: load i32*, i32** %0
; CHECK-SAME: !align ![[ALIGN:[0-9]+]]
define void @test_load_load_combine_metadata(i32**, i32**, i32**) {
  %a = load i32*, i32** %0, !align !0
  %b = load i32*, i32** %0, !align !1
  store i32 0, i32* %a
  store i32 0, i32* %b
  ret void
}

; CHECK: ![[ALIGN]] = !{i64 4}

!0 = !{i64 4}
!1 = !{i64 8}