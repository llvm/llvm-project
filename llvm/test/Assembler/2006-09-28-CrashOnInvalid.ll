; Test for PR902.  This program is erroneous, but should not crash llvm-as.
; This tests that a simple error is caught and processed correctly.
; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: floating point constant invalid for type

define void @test() {
  add i32 1, 2.0
  ret void
}
