; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=prescott | FileCheck %s --check-prefix=PRESCOTT
; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=nehalem | FileCheck %s --check-prefix=NEHALEM

;;; TODO: (1) Some of the loads and stores are certainly unaligned and (2) the first load and first
;;; store overlap with the second load and second store respectively.
;;;
;;; Is either of these sequences ideal? 

define float @foo(i8* nocapture %buf, float %a, float %b) nounwind uwtable {
; PRESCOTT-LABEL: foo:
; PRESCOTT:       # %bb.0: # %entry
; PRESCOTT-NEXT:    movq   .Ltmp0+14(%rip), %rax
; PRESCOTT-NEXT:    movq   %rax, 14(%rdi)
; PRESCOTT-NEXT:    movq   .Ltmp0+8(%rip), %rax
; PRESCOTT-NEXT:    movq   %rax, 8(%rdi)
; PRESCOTT-NEXT:    movq   .Ltmp0(%rip), %rax
; PRESCOTT-NEXT:    movq   %rax, (%rdi)
;
; NEHALEM-LABEL: foo:
; NEHALEM:       # %bb.0: # %entry
; NEHALEM-NEXT:    movq .Ltmp0+14(%rip), %rax
; NEHALEM-NEXT:    movq %rax, 14(%rdi)
; NEHALEM-NEXT:    movups .Ltmp0(%rip), %xmm2
; NEHALEM-NEXT:    movups %xmm2, (%rdi)

entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* blockaddress(@foo, %out), i64 22, i32 1, i1 false)
  br label %out

out:                                              ; preds = %entry
  %add = fadd float %a, %b
  ret float %add
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
