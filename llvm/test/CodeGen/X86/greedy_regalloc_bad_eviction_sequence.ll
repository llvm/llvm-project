; RUN: llc < %s -march=x86 -regalloc=greedy -stop-after=greedy | FileCheck %s
; Make sure bad eviction sequence doesnt occur

; Part of the fix for bugzilla 26810.
; This test is meant to make sure bad eviction sequence like the one described
; below does not occur
;
; movl	%ebp, 8($esp)           # 4-byte Spill
; movl	%ecx, %ebp
; movl	%ebx, %ecx
; movl	$edi, %ebx
; movl	$edx, $edi
; cltd
; idivl	%esi
; movl	$edi, $edx
; movl	%ebx, $edi
; movl	%ecx, %ebx
; movl	%ebp, %ecx
; movl	16($esp), %ebp          # 4 - byte Reload

; Make sure we have no redundant copies in the problematic code seqtion
; CHECK-LABEL: name: bar
; CHECK: bb.3.for.body:
; CHECK: $eax = COPY
; CHECK-NEXT: CDQ
; CHECK-NEXT: IDIV32r
; CHECK-NEXT: ADD32rr


target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-linux-gnu"


; Function Attrs: norecurse nounwind readonly
define i32 @bar(i32 %size, i32* nocapture readonly %arr, i32* nocapture readnone %tmp) local_unnamed_addr #1 {
entry:
  %0 = load i32, i32* %arr, align 4, !tbaa !3
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i32 1
  %1 = load i32, i32* %arrayidx3, align 4, !tbaa !3
  %arrayidx5 = getelementptr inbounds i32, i32* %arr, i32 2
  %2 = load i32, i32* %arrayidx5, align 4, !tbaa !3
  %arrayidx7 = getelementptr inbounds i32, i32* %arr, i32 3
  %3 = load i32, i32* %arrayidx7, align 4, !tbaa !3
  %arrayidx9 = getelementptr inbounds i32, i32* %arr, i32 4
  %4 = load i32, i32* %arrayidx9, align 4, !tbaa !3
  %arrayidx11 = getelementptr inbounds i32, i32* %arr, i32 5
  %5 = load i32, i32* %arrayidx11, align 4, !tbaa !3
  %arrayidx13 = getelementptr inbounds i32, i32* %arr, i32 6
  %6 = load i32, i32* %arrayidx13, align 4, !tbaa !3
  %arrayidx15 = getelementptr inbounds i32, i32* %arr, i32 7
  %7 = load i32, i32* %arrayidx15, align 4, !tbaa !3
  %arrayidx17 = getelementptr inbounds i32, i32* %arr, i32 8
  %8 = load i32, i32* %arrayidx17, align 4, !tbaa !3
  %cmp69 = icmp sgt i32 %size, 1
  br i1 %cmp69, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %x0.0.lcssa = phi i32 [ %0, %entry ], [ %add, %for.body ]
  %x1.0.lcssa = phi i32 [ %1, %entry ], [ %sub, %for.body ]
  %x2.0.lcssa = phi i32 [ %2, %entry ], [ %mul, %for.body ]
  %x3.0.lcssa = phi i32 [ %3, %entry ], [ %div, %for.body ]
  %x4.0.lcssa = phi i32 [ %4, %entry ], [ %add19, %for.body ]
  %x5.0.lcssa = phi i32 [ %5, %entry ], [ %sub20, %for.body ]
  %x6.0.lcssa = phi i32 [ %6, %entry ], [ %add21, %for.body ]
  %x7.0.lcssa = phi i32 [ %7, %entry ], [ %mul22, %for.body ]
  %x8.0.lcssa = phi i32 [ %8, %entry ], [ %sub23, %for.body ]
  %mul24 = mul nsw i32 %x1.0.lcssa, %x0.0.lcssa
  %mul25 = mul nsw i32 %mul24, %x2.0.lcssa
  %mul26 = mul nsw i32 %mul25, %x3.0.lcssa
  %mul27 = mul nsw i32 %mul26, %x4.0.lcssa
  %mul28 = mul nsw i32 %mul27, %x5.0.lcssa
  %mul29 = mul nsw i32 %mul28, %x6.0.lcssa
  %mul30 = mul nsw i32 %mul29, %x7.0.lcssa
  %mul31 = mul nsw i32 %mul30, %x8.0.lcssa
  ret i32 %mul31

for.body:                                         ; preds = %entry, %for.body
  %i.079 = phi i32 [ %inc, %for.body ], [ 1, %entry ]
  %x8.078 = phi i32 [ %sub23, %for.body ], [ %8, %entry ]
  %x7.077 = phi i32 [ %mul22, %for.body ], [ %7, %entry ]
  %x6.076 = phi i32 [ %add21, %for.body ], [ %6, %entry ]
  %x5.075 = phi i32 [ %sub20, %for.body ], [ %5, %entry ]
  %x4.074 = phi i32 [ %add19, %for.body ], [ %4, %entry ]
  %x3.073 = phi i32 [ %div, %for.body ], [ %3, %entry ]
  %x2.072 = phi i32 [ %mul, %for.body ], [ %2, %entry ]
  %x1.071 = phi i32 [ %sub, %for.body ], [ %1, %entry ]
  %x0.070 = phi i32 [ %add, %for.body ], [ %0, %entry ]
  %add = add nsw i32 %x1.071, %x0.070
  %sub = sub nsw i32 %x1.071, %x2.072
  %mul = mul nsw i32 %x3.073, %x2.072
  %div = sdiv i32 %x3.073, %x4.074
  %add19 = add nsw i32 %x5.075, %x4.074
  %sub20 = sub nsw i32 %x5.075, %x6.076
  %add21 = add nsw i32 %x7.077, %x6.076
  %mul22 = mul nsw i32 %x8.078, %x7.077
  %sub23 = sub nsw i32 %x8.078, %add
  %inc = add nuw nsw i32 %i.079, 1
  %exitcond = icmp eq i32 %inc, %size
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !7
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{!"clang version 5.0.0 (cfe/trunk 305640)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}
