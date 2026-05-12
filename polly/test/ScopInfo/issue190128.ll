; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s 2>&1 | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/190128
;
; void func(int arg, unsigned char arr_4[11]) {
;     int shl1 = 2147483592LL << arg; // 2147483592
;     int trunc1 = (short)(shl1); // -56
;     int start = trunc1 + 56; // = 0
;     for (short i_0 = start; i_0 < 1; i_0 += 1) {
;        arr_4[i_0] = i_0;
;        for (int i_2 = 1; i_2 < 3; i_2 += 1)
;          ; // somehow this matters -- different CFG
;     }
; }
;
; The constraint -58 <= shl < 32768 (ignorable trunc range) must be checked in an RTC (here: InvalidConstant).
; Alternatively, %conv6 could be used as a parameter, instead of %shl.

; CHECK:      Context:
; CHECK-NEXT: [shl] -> {  : -9223372036854775808 <= shl <= 9223372036854775800 }
; CHECK:      Assumed Context:
; CHECK-NEXT: [shl] -> {  :  }
; CHECK:      Invalid Context:
; CHECK-NEXT: [shl] -> { : shl <= -57 or shl >= 32768 }
; CHECK:      Defined Behavior Context:
; CHECK-NEXT: [shl] -> {  : -56 <= shl <= 32710 }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @func(i32 noundef %arg, ptr noundef writeonly captures(none) %arr_4) local_unnamed_addr #0 {
entry:
  %sh_prom = zext nneg i32 %arg to i64
  %shl = shl i64 2147483592, %sh_prom
  %conv1 = trunc i64 %shl to i16
  %add = add i16 %conv1, 56
  %cmp22 = icmp slt i16 %add, 1
  br i1 %cmp22, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i_0.023 = phi i16 [ %add15, %for.body ], [ %add, %entry ]
  %conv6 = trunc i16 %i_0.023 to i8
  %idxprom = sext i16 %i_0.023 to i64
  %arrayidx = getelementptr inbounds i8, ptr %arr_4, i64 %idxprom
  store i8 %conv6, ptr %arrayidx, align 1, !tbaa !8
  %add15 = add nsw i16 %i_0.023, 1
  %cmp = icmp ugt i16 %i_0.023, 32766
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !9
}

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.errno.tbaa = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 23.0.0git (/home/meinersbur/src/llvm/polly/_src/clang a2d3783b451c0c19a5eb09b1ab9a1c66d81ab6ca)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!6, !6, i64 0}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
