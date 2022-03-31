; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @test_load_load_combine_metadata(
; Check that dereferenceable_or_null metadata is combined
; CHECK: load i32*, i32** %0
; CHECK-SAME: !dereferenceable_or_null ![[DEREF:[0-9]+]]
define void @test_load_load_combine_metadata(i32**, i32**, i32**) {
  %a = load i32*, i32** %0, !dereferenceable_or_null !0
  %b = load i32*, i32** %0, !dereferenceable_or_null !1
  store i32 0, i32* %a
  store i32 0, i32* %b
  ret void
}

; CHECK: ![[DEREF]] = !{i64 4}

!0 = !{i64 4}
!1 = !{i64 8}
