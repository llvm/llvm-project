; The pass here doesn't matter (we use deadargelim), but test
; that the -run-twice options exists, generates output, and
; doesn't crash
; RUN: opt -run-twice -passes=deadargelim -S < %s | FileCheck %s

; CHECK: define internal void @test
define internal {} @test() {
  ret {} undef
}

define void @caller() {
  call {} @test()
  ret void
}
