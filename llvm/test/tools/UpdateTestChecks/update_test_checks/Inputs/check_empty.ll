; RUN: opt < %s -S | FileCheck %s

; Test whether UTC checks empty lines instead of skipping them.
define i32 @test(i32 %x) {
entry:
  br label %block1

block1:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %block2, label %exit1

block2:
  br i1 %cmp, label %block3, label %exit2

block3:
  br i1 %cmp, label %exit3, label %exit4

exit1:
  ret i32 0

exit2:
  ret i32 %x

exit3:
  ret i32 %x

exit4:
  ret i32 %x
}
