; RUN: llc < %s | FileCheck %s

target triple = "i686-unknown-unknown"

define i32 @test(i32 %X) {
; CHECK-LABEL: test:
entry:
  ret i32 %X
; CHECK-NOT: subl %esp
}
