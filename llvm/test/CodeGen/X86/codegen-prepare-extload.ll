; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win64 | FileCheck %s
; rdar://7304838

; CodeGenPrepare should move the zext into the block with the load
; so that SelectionDAG can select it with the load.

; CHECK: movzbl ({{%rdi|%rcx}}), %eax

define void @foo(i8* %p, i32* %q) {
entry:
  %t = load i8* %p
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %t to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}
