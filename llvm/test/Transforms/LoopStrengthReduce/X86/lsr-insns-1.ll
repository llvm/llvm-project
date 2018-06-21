; RUN: opt < %s -loop-reduce -mtriple=x86_64  -S | FileCheck %s -check-prefix=BOTH -check-prefix=INSN
; RUN: opt < %s -loop-reduce -mtriple=x86_64 -lsr-insns-cost=false -S | FileCheck %s -check-prefix=BOTH -check-prefix=REGS
; RUN: llc < %s -O2 -march=x86-64 -lsr-insns-cost -asm-verbose=0 | FileCheck %s

; OPT test checks that LSR optimize compare for static counter to compare with 0.

; BOTH: for.body:
; INSN: icmp eq i64 %lsr.iv.next, 0
; REGS: icmp eq i64 %indvars.iv.next, 1024

; LLC test checks that LSR optimize compare for static counter.
; That means that instead of creating the following:
;   movl %ecx, (%rdx,%rax,4)
;   incq %rax
;   cmpq $1024, %rax
; LSR should optimize out cmp:
;   movl %ecx, 4096(%rdx,%rax)
;   addq $4, %rax
; or
;   movl %ecx, 4096(%rdx,%rax,4)
;   incq %rax

; CHECK:      LBB0_1:
; CHECK-NEXT:   movl 4096(%{{.+}},[[REG:%[0-9a-z]+]]
; CHECK-NEXT:   addl 4096(%{{.+}},[[REG]]
; CHECK-NEXT:   movl %{{.+}}, 4096(%{{.+}},[[REG]]
; CHECK-NOT:    cmp
; CHECK:        jne

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: norecurse nounwind uwtable
define void @foo(i32* nocapture readonly %x, i32* nocapture readonly %y, i32* nocapture %q) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %y, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %tmp1, %tmp
  %arrayidx4 = getelementptr inbounds i32, i32* %q, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
