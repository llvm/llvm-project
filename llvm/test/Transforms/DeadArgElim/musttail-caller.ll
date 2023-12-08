; RUN: opt -passes=deadargelim -S < %s | FileCheck %s
; PR36441
; Dead arguments should not be removed in presence of `musttail` calls.

; CHECK-LABEL: define internal void @test(i32 %a, i32 %b)
; CHECK: musttail call void @foo(i32 %a, i32 0)
; FIXME: we should replace those with `undef`s
define internal void @test(i32 %a, i32 %b) {
  musttail call void @foo(i32 %a, i32 0)
  ret void
}

; CHECK-LABEL: define internal void @foo(i32 %a, i32 %b)
define internal void @foo(i32 %a, i32 %b) {
  ret void
}
