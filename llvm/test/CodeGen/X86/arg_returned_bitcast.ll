; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s

; Test that the "returned" attribute "works" even if there is a bitcast between
; the argument and return value.

declare ptr @bar(ptr returned)

define ptr @foo(ptr) {
  %r = tail call ptr @bar(ptr %0)
; CHECK: jmp    bar
  ret ptr %r
}
