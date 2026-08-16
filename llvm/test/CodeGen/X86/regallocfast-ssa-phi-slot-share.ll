; RUN: llc -mtriple=x86_64-linux -O0 -regalloc-fast-ssa -verify-machineinstrs < %s | FileCheck %s

;; A PHI source defined in the incoming predecessor and dying there takes the
;; destination's stack slot, so its spill is the edge transfer and no copy is
;; needed. That spill lands before the end of the predecessor, so nothing may
;; read the destination's slot in between -- a PHI in a successor does, on the
;; edge.
;;
;; Offsets are spelled out in the checks on purpose: which slot each store
;; targets is the property under test, and update_llc_test_checks.py wildcards
;; spill offsets unconditionally, so do not regenerate this file with it.

;; Three edge copies (no sharing):
;;   %k1 into %k: %k is read in %latch, by the add.
;;   %s into %d: %o reads %d on the %latch -> %head edge, which happens in
;;      %latch, after a shared spill of %s would have overwritten %d's slot.
;;   %d into %o: %d is not defined in %latch.
;; The copy reading %d is ordered ahead of the one writing it.
define i32 @phi_reads_dst_on_edge(i32 %n) nounwind {
; CHECK-LABEL: phi_reads_dst_on_edge:
; CHECK:         movl $100, %eax
; CHECK-NEXT:    movl %eax, -8(%rsp)
; CHECK:       # %latch
; CHECK:         movl -8(%rsp), %ecx
; CHECK-NEXT:    movl -4(%rsp), %edx
; CHECK-NEXT:    addl $1, %edx
; CHECK-NEXT:    movl %edx, %eax
; CHECK-NEXT:    addl $200, %eax
; CHECK-NEXT:    movl %edx, -4(%rsp)
; CHECK-NEXT:    movl %ecx, -12(%rsp)
; CHECK-NEXT:    movl %eax, -8(%rsp)
; CHECK:       # %exit
; CHECK-NEXT:    movl -12(%rsp), %eax
; CHECK-NEXT:    retq
entry:
  br label %head

head:
  %k = phi i32 [ 0,   %entry ], [ %k1, %latch ]
  %d = phi i32 [ 100, %entry ], [ %s,  %latch ]
  %o = phi i32 [ -1,  %entry ], [ %d,  %latch ]
  %c = icmp slt i32 %k, %n
  br i1 %c, label %latch, label %exit

latch:
  %k1 = add i32 %k, 1
  %s = add i32 %k1, 200
  br label %head

exit:
  ret i32 %o
}

;; Same loop without %o. Nothing reads %d in %latch or on the edge out of it, so
;; %s takes %d's slot and the add of $200 stores straight into it -- no edge
;; copy for %d. %k1 still cannot take %k's slot, %k being read in %latch, so a
;; single copy remains.
define i32 @share_across_back_edge(i32 %n) nounwind {
; CHECK-LABEL: share_across_back_edge:
; CHECK:         movl $100, %eax
; CHECK-NEXT:    movl %eax, -8(%rsp)
; CHECK:       # %latch
; CHECK:         movl -4(%rsp), %eax
; CHECK-NEXT:    addl $1, %eax
; CHECK-NEXT:    movl %eax, %ecx
; CHECK-NEXT:    addl $200, %ecx
; CHECK-NEXT:    movl %ecx, -8(%rsp)
; CHECK-NEXT:    movl %eax, -4(%rsp)
; CHECK:       # %exit
; CHECK-NEXT:    movl -8(%rsp), %eax
; CHECK-NEXT:    retq
entry:
  br label %head

head:
  %k = phi i32 [ 0,   %entry ], [ %k1, %latch ]
  %d = phi i32 [ 100, %entry ], [ %s,  %latch ]
  %c = icmp slt i32 %k, %n
  br i1 %c, label %latch, label %exit

latch:
  %k1 = add i32 %k, 1
  %s = add i32 %k1, 200
  br label %head

exit:
  ret i32 %d
}
