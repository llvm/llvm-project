; RUN: llc -mtriple=hexagon -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s

; Test that the compiler deallocates the register saved area on Linux
; for functions that do not need a frame pointer.

; CHECK: r29 = add(r29,#-[[SIZE:[0-9]+]]
; CHECK: r29 = add(r29,#[[SIZE]])

define void @test(...) {
entry:
  ret void
}

