; RUN: opt -passes='loop-vectorize' -debug -S < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; This checks we don't crash when the inner loop we're trying to vectorize
; is a SCEV AddRec with respect to an outer loop.

; In this case, the problematic PHI is:
; %0 = phi i32 [ undef, %for.cond1.preheader ], [ %inc54, %for.body3 ]
; Since %inc54 is the IV of the outer loop, and %0 equivalent to it,
; we get the situation described above.

; Code that leads to this situation can look something like:
;
; int a, b[1], c;
; void fn1 ()
; {
;  for (; c; c++)
;    for (a = 0; a; a++)
;      b[c] = 4;
; }
;
; The PHI is an artifact of the register promotion of c.

; Note that we can no longer get the vectorizer to actually see such PHIs,
; because LV now simplifies the loop internally, but the test is still
; useful as a regression test, and in case loop-simplify behavior changes.

@c = external global i32, align 4
@a = external global i32, align 4
@b = external global [1 x i32], align 4

; We can vectorize this loop because we are storing an invariant value into an
; invariant address.

; CHECK: LV: We can vectorize this loop!
; CHECK-LABEL: @test
define void @test() {
entry:
  %a.promoted2 = load i32, i32* @a, align 1
  %c.promoted = load i32, i32* @c, align 1
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.for.inc4_crit_edge, %entry
  %inc54 = phi i32 [ %inc5, %for.cond1.for.inc4_crit_edge ], [ %c.promoted, %entry ]
  %inc.lcssa3 = phi i32 [ %inc.lcssa, %for.cond1.for.inc4_crit_edge ], [ %a.promoted2, %entry ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %inc1 = phi i32 [ %inc.lcssa3, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %0 = phi i32 [ undef, %for.cond1.preheader ], [ %inc54, %for.body3 ]
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [1 x i32], [1 x i32]* @b, i64 0, i64 %idxprom
  store i32 4, i32* %arrayidx, align 4
  %inc = add nsw i32 %inc1, 1
  %tobool2 = icmp eq i32 %inc, 0
  br i1 %tobool2, label %for.cond1.for.inc4_crit_edge, label %for.body3

for.cond1.for.inc4_crit_edge:                     ; preds = %for.body3
  %inc.lcssa = phi i32 [ %inc, %for.body3 ]
  %.lcssa = phi i32 [ %inc54, %for.body3 ]
  %inc5 = add nsw i32 %.lcssa, 1
  br label %for.cond1.preheader
}
