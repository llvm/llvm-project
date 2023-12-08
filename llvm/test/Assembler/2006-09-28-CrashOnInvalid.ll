; Test for PR902.  This program is erroneous, but should not crash llvm-as.
; This tests that a simple error is caught and processed correctly.
; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: grep "floating point constant invalid for type" %t

define void @test() {
  add i32 1, 2.0
  ret void
}
