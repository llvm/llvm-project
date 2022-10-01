; RUN: opt < %s -loop-reduce -mtriple=x86_64-- -S | FileCheck %s -check-prefix=BOTH -check-prefix=INSN
; RUN: opt < %s -loop-reduce -mtriple=x86_64-- -lsr-insns-cost=false -S | FileCheck %s -check-prefix=BOTH -check-prefix=REGS
; RUN: llc < %s -O2 -mtriple=x86_64-- -lsr-insns-cost -asm-verbose=0 | FileCheck %s

; OPT checks that LSR prefers less instructions to less registers.
; For x86 LSR should prefer complicated address to new lsr induction
; variables.

; BOTH: for.body:
; INSN:   getelementptr i32, i32* %x, i64 %indvars.iv
; INSN:   getelementptr i32, i32* %y, i64 %indvars.iv
; INSN:   getelementptr i32, i32* %q, i64 %indvars.iv
; REGS:   %lsr.iv4 = phi
; REGS:   %lsr.iv2 = phi
; REGS:   %lsr.iv1 = phi
; REGS:   getelementptr i32, i32* %lsr.iv1, i64 1
; REGS:   getelementptr i32, i32* %lsr.iv2, i64 1
; REGS:   getelementptr i32, i32* %lsr.iv4, i64 1

; LLC checks that LSR prefers less instructions to less registers.
; LSR should prefer complicated address to additonal add instructions.

; CHECK:      LBB0_2:
; CHECK-NEXT:   movl (%r{{.+}},{{.*}}), [[REG:%[a-z0-9]+]]
; CHECK-NEXT:   addl (%r{{.+}},{{.*}}), [[REG]]
; CHECK-NEXT:   movl [[REG]], (%{{.*}})

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: norecurse nounwind uwtable
define void @foo(i32* nocapture readonly %x, i32* nocapture readonly %y, i32* nocapture %q, i32 %n) {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %y, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %tmp1, %tmp
  %arrayidx4 = getelementptr inbounds i32, i32* %q, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
