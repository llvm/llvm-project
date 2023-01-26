; RUN: opt < %s -passes=indvars -S | FileCheck %s
; PR11350: Check that SimplifyIndvar handles a cycle of useless self-phis.

; CHECK-LABEL: @test(
; CHECK-NOT: lcssa = phi
define void @test() nounwind {
entry:
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  br label %for.cond.outer

for.cond.outer:                                   ; preds = %for.cond.preheader, %for.end
  %p_41.addr.0.ph = phi i32 [ %p_41.addr.1.lcssa, %for.end ], [ 1, %for.cond.preheader ]
  br label %for.cond

for.cond:
  br i1 true, label %for.end, label %for.ph

for.ph:                                   ; preds = %for.cond4.preheader
  br label %for.end

for.end:
  %p_41.addr.1.lcssa = phi i32 [ undef, %for.ph ], [ %p_41.addr.0.ph, %for.cond ]
  %p_68.lobit.i = lshr i32 %p_41.addr.1.lcssa, 31
  %cmp7 = icmp eq i32 %p_41.addr.1.lcssa, 0
  %conv8 = zext i1 %cmp7 to i32
  br label %for.cond.outer
}
