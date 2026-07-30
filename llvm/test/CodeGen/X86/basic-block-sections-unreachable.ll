; Check that basic block section is emitted when a non-entry block has no predecessors.
; RUN: llc < %s -mtriple=x86_64 -O0 -basic-block-sections=all | FileCheck %s --check-prefix=CHECK-SECTIONS
; RUN: llc < %s -mtriple=x86_64 -O0 | FileCheck %s --check-prefix=CHECK-NOSECTIONS
define void @foo(ptr %bar) {
  %v = load i32, ptr %bar
  switch i32 %v, label %default [
    i32 0, label %target
  ]
target:
  ret void
;; This is the block which will not have any predecessors. If the block is not garbage collected, it must
;; be placed in a basic block section with a corresponding symbol.
default:
  unreachable
; CHECK-NOSECTIONS:     # %bb.2:     # %default
; CHECK-SECTIONS:       .section .text.foo,"ax",@progbits,unique,2
; CHECK-SECTIONS-NEXT:  foo.__part.2:       # %default
}
