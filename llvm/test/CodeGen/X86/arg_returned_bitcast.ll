; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s

; Test that the "returned" attribute "works" even if there is a bitcast between
; the argument and return value.

declare double* @bar(i8* returned)

define double* @foo(i8*) {
  %r = tail call double* @bar(i8* %0)
; CHECK: jmp    bar
  ret double* %r
}
