; RUN: llc < %s -mtriple=x86_64-uefi | FileCheck %s -check-prefix=UEFIFAST64
; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s -check-prefix=UEFIFAST64

declare fastcc i32 @fastcallee1(i32 %a1, i32 %a2, i32 %a3, i32 %a4)

define fastcc i32 @fastcaller1(i32 %in1, i32 %in2) nounwind {
;; Test that the caller allocates stack space for callee to spill the register arguments.
; UEFIFAST64-LABEL: fastcaller1:
; UEFIFAST64:       # %bb.0: # %entry
; UEFIFAST64-NEXT:    subq	$40, %rsp
; UEFIFAST64-NEXT:    movl %ecx, %r8d
; UEFIFAST64-NEXT:    movl %edx, %r9d
; UEFIFAST64-NEXT:    callq fastcallee1
; UEFIFAST64-NEXT:    addq	$40, %rsp
; UEFIFAST64-NEXT:    retq
entry:
  %tmp11 = call fastcc i32 @fastcallee1(i32 %in1, i32 %in2, i32 %in1, i32 %in2)
  ret i32 %tmp11
}
