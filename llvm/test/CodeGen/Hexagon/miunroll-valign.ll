; RUN: llc -O3  -march=hexagon -enable-machine-unroller=true < %s | FileCheck %s
; REQUIRES: asserts

; This test used to assert because machine unroller copied valign instruction
; with the third parameter as IntRegs instead of IntRegsLow8. Instruction valign
; requires its third parameter to have IntRegsLow8 type.

; CHECK: loop0(
; CHECK: valign
; CHECK: valign
; CHECK: endloop0

@a = common dso_local local_unnamed_addr global i32 0, align 4
@b = common dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nosync nounwind
define dso_local i32 @c(i32 %e_init, <32 x i32> %k_init, <32 x i32> %v0, <32 x i32> %v1, <32 x i32> %v2) #0 {
entry:
  %0 = load i32, ptr @b, align 4
  %1 = load i32, ptr @a, align 4
  %2 = icmp slt i32 %1, %0
  %cmp = icmp ult ptr inttoptr (i32 2 to ptr), @c
  %or.cond = select i1 %cmp , i1 %2, i1 false
  br i1 %or.cond, label %for.cond1.preheader.us.preheader, label %for.end3

for.cond1.preheader.us.preheader:                 ; preds = %entry
  %3 = shl i32 %0, 2
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond.loopexit.us, %for.cond1.preheader.us.preheader
  %4 = phi i32 [ %9, %for.cond.loopexit.us ], [ %1, %for.cond1.preheader.us.preheader ]
  %e.019.us = phi i32 [ %e.1.lcssa.us, %for.cond.loopexit.us ], [ %e_init, %for.cond1.preheader.us.preheader ]
  %k.018.us = phi <32 x i32> [ %k.1.lcssa.us, %for.cond.loopexit.us ], [ %k_init, %for.cond1.preheader.us.preheader ]
  %cmp14.us = icmp slt i32 %4, %0
  br i1 %cmp14.us, label %for.body2.us.preheader, label %for.cond.loopexit.us

for.body2.us.preheader:                           ; preds = %for.cond1.preheader.us
  %.neg = mul i32 %4, -4
  br label %for.body2.us

for.body2.us:                                     ; preds = %for.body2.us, %for.body2.us.preheader
  %5 = phi i32 [ %inc.us, %for.body2.us ], [ %4, %for.body2.us.preheader ]
  %e.116.us = phi i32 [ %add.us, %for.body2.us ], [ %e.019.us, %for.body2.us.preheader ]
  %k.115.us = phi <32 x i32> [ %8, %for.body2.us ], [ %k.018.us, %for.body2.us.preheader ]
  %6 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v0, <32 x i32> %v1, i32 %e.116.us)
  %add.us = add nsw i32 %e.116.us, 4
  %7 = tail call <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32> %6, <32 x i32> %v2)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32> %k.115.us, <32 x i32> %7)
  %inc.us = add nsw i32 %5, 1
  %exitcond.not = icmp eq i32 %inc.us, %0
  br i1 %exitcond.not, label %for.cond1.for.cond.loopexit_crit_edge.us, label %for.body2.us

for.cond.loopexit.us:                             ; preds = %for.cond1.for.cond.loopexit_crit_edge.us, %for.cond1.preheader.us
  %9 = phi i32 [ %0, %for.cond1.for.cond.loopexit_crit_edge.us ], [ %4, %for.cond1.preheader.us ]
  %k.1.lcssa.us = phi <32 x i32> [ %8, %for.cond1.for.cond.loopexit_crit_edge.us ], [ %k.018.us, %for.cond1.preheader.us ]
  %e.1.lcssa.us = phi i32 [ %11, %for.cond1.for.cond.loopexit_crit_edge.us ], [ %e.019.us, %for.cond1.preheader.us ]
  %cmp2 = icmp ult ptr inttoptr (i32 2 to ptr), @c
  br i1 %cmp2, label %for.cond1.preheader.us, label %for.end3

for.cond1.for.cond.loopexit_crit_edge.us:         ; preds = %for.body2.us
  %10 = add i32 %3, %e.019.us
  %11 = add i32 %.neg, %10
  store i32 %0, ptr @a, align 4
  br label %for.cond.loopexit.us

for.end3:                                         ; preds = %for.cond.loopexit.us, %entry
  %k.0.lcssa = phi <32 x i32> [ %k_init, %entry ], [ %k.1.lcssa.us, %for.cond.loopexit.us ]
  store <32 x i32> %k.0.lcssa, ptr inttoptr (i32 2 to ptr), align 128
  ret i32 %e_init
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vmpy.qf32.sf.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nofree nosync nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv68" "target-features"="+hvx-length128b,+hvxv68,+v68,-long-calls" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
