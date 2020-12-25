;; This test verifies that basic-block-sections works with address-taken basic blocks.
; RUN: llc < %s -mtriple=x86_64 -basic-block-sections=all | FileCheck %s

define void @foo(i1 zeroext %0) nounwind {
entry:
  %1 = select i1 %0, i8* blockaddress(@foo, %bb1), i8* blockaddress(@foo, %bb2) ; <i8*> [#uses=1]
  indirectbr i8* %1, [label %bb1, label %bb2]

; CHECK:         .text
; CHECK-LABEL: foo:
; CHECK:         movl    $.Ltmp0, %eax
; CHECK-NEXT:    movl    $.Ltmp1, %ecx
; CHECK-NEXT:    cmovneq %rax, %rcx
; CHECK-NEXT:    jmpq    *%rcx

bb1:                                                ; preds = %entry
  %2 = call i32 @bar()
  ret void
; CHECK:         .section .text,"ax",@progbits,unique,1
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:  foo.__part.1
; CHECK-NEXT:    callq   bar
;

bb2:                                                ; preds = %entry
  %3 = call i32 @baz()
  ret void
; CHECK:         .section .text,"ax",@progbits,unique,2
; CHECK-NEXT:  .Ltmp1:
; CHECK-NEXT:  foo.__part.2
; CHECK-NEXT:    callq   baz
}

declare i32 @bar()
declare i32 @baz()
