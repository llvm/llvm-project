; RUN: opt -S -dxil-prepare < %s | FileCheck %s
target triple = "dxil-unknown-unknown"

; Make sure not crash when has typed ptr.
; CHECK:@test

define i64 @test(i64* %p) {
  %v = load i64, i64* %p
  ret i64 %v
}
