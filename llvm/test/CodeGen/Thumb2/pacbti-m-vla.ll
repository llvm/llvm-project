; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; int g(int, int *);
;
; int f(int n) {
;   int a[n];
;   g(n, a);
;   int s = 0;
;   for (int i = 0; i < n; ++i)
;     s += a[i];
;   return s;
; }

define hidden i32 @f(i32 %n) local_unnamed_addr #0 {
entry:
  %vla = alloca i32, i32 %n, align 4
  %call = call i32 @g(i32 %n, ptr nonnull %vla) #0
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = add i32 %n, -1
  %xtraiter = and i32 %n, 3
  %1 = icmp ult i32 %0, 3
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i32 %n, -4
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %add.lcssa.ph = phi i32 [ undef, %for.body.preheader ], [ %add.3, %for.body ]
  %i.010.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.3, %for.body ]
  %s.09.unr = phi i32 [ 0, %for.body.preheader ], [ %add.3, %for.body ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa
  %arrayidx.epil = getelementptr inbounds i32, ptr %vla, i32 %i.010.unr
  %2 = load i32, ptr %arrayidx.epil, align 4
  %add.epil = add nsw i32 %2, %s.09.unr
  %epil.iter.cmp.not = icmp eq i32 %xtraiter, 1
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.body.epil.1

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil.2, %for.body.epil.1, %for.body.epil, %entry
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %add.epil, %for.body.epil ], [ %add.epil.1, %for.body.epil.1 ], [ %add.epil.2, %for.body.epil.2 ]
  ret i32 %s.0.lcssa

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.010 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.3, %for.body ]
  %s.09 = phi i32 [ 0, %for.body.preheader.new ], [ %add.3, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.3, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %vla, i32 %i.010
  %3 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %3, %s.09
  %inc = or disjoint i32 %i.010, 1
  %arrayidx.1 = getelementptr inbounds i32, ptr %vla, i32 %inc
  %4 = load i32, ptr %arrayidx.1, align 4
  %add.1 = add nsw i32 %4, %add
  %inc.1 = or disjoint i32 %i.010, 2
  %arrayidx.2 = getelementptr inbounds i32, ptr %vla, i32 %inc.1
  %5 = load i32, ptr %arrayidx.2, align 4
  %add.2 = add nsw i32 %5, %add.1
  %inc.2 = or disjoint i32 %i.010, 3
  %arrayidx.3 = getelementptr inbounds i32, ptr %vla, i32 %inc.2
  %6 = load i32, ptr %arrayidx.3, align 4
  %add.3 = add nsw i32 %6, %add.2
  %inc.3 = add nuw nsw i32 %i.010, 4
  %niter.nsub.3 = add i32 %niter, -4
  %niter.ncmp.3 = icmp eq i32 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body

for.body.epil.1:                                  ; preds = %for.body.epil
  %inc.epil = add nuw nsw i32 %i.010.unr, 1
  %arrayidx.epil.1 = getelementptr inbounds i32, ptr %vla, i32 %inc.epil
  %7 = load i32, ptr %arrayidx.epil.1, align 4
  %add.epil.1 = add nsw i32 %7, %add.epil
  %epil.iter.cmp.1.not = icmp eq i32 %xtraiter, 2
  br i1 %epil.iter.cmp.1.not, label %for.cond.cleanup, label %for.body.epil.2

for.body.epil.2:                                  ; preds = %for.body.epil.1
  %inc.epil.1 = add nuw nsw i32 %i.010.unr, 2
  %arrayidx.epil.2 = getelementptr inbounds i32, ptr %vla, i32 %inc.epil.1
  %8 = load i32, ptr %arrayidx.epil.2, align 4
  %add.epil.2 = add nsw i32 %8, %add.epil.1
  br label %for.cond.cleanup
}

; CHECK-LABEL: f:
; CHECK:       pac    r12, lr, sp
; CHECK-NEXT: .save   {r4, r5, r6, r7, lr}
; CHECK-NEXT: push    {r4, r5, r6, r7, lr}
; CHECK-NEXT: .cfi_def_cfa_offset 20
; CHECK-NEXT: .cfi_offset lr, -4
; CHECK-NEXT: .cfi_offset r7, -8
; CHECK-NEXT: .cfi_offset r6, -12
; CHECK-NEXT: .cfi_offset r5, -16
; CHECK-NEXT: .cfi_offset r4, -20
; CHECK-NEXT: .setfp r7, sp, #12
; CHECK-NEXT: add    r7, sp, #12
; CHECK-NEXT: .cfi_def_cfa r7, 8
; CHECK-NEXT: .save    {r8, r9, ra_auth_code}
; CHECK-NEXT: push.w   {r8, r9, r12}
; CHECK-NEXT: .cfi_offset ra_auth_code, -24
; CHECK-NEXT: .cfi_offset r9, -28
; CHECK-NEXT: .cfi_offset r8, -32
; ...
; CHECK:      sub.w  r[[N:[0-9]*]], r7, #24
; CHECK-NEXT: mov    sp, r[[N]]
; CHECK-NEXT: pop.w  {r8, r9, r12}
; CHECK-NEXT: pop.w  {r4, r5, r6, r7, lr}
; CHECK-NEXT: aut    r12, lr, sp
; CHECK-NEXT: bx     lr

declare dso_local i32 @g(i32, ptr) local_unnamed_addr #0

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
