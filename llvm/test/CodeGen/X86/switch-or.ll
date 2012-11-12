; RUN: llc -march=x86 -asm-verbose=false < %s | FileCheck %s

; Check that merging switch cases that differ in one bit works.
; CHECK: orl $2
; CHECK-NEXT: cmpl $6

define void @foo(i32 %variable) nounwind {
entry:
  switch i32 %variable, label %if.end [
    i32 4, label %if.then
    i32 6, label %if.then
  ]

if.then:
  %call = tail call i32 (...)* @bar() nounwind
  ret void

if.end:
  ret void
}

declare i32 @bar(...) nounwind
