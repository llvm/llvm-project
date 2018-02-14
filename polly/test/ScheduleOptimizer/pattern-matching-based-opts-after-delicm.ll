; RUN: opt %loadPolly \
; RUN: -polly-pattern-matching-based-opts=true \
; RUN: -polly-optree -polly-delicm -polly-simplify \
; RUN: -polly-opt-isl -debug < %s 2>&1 \
; RUN: | FileCheck %s
; REQUIRES: asserts

; Check that the pattern matching detects the matrix multiplication pattern
; after a full run of -polly-optree and -polly-delicm, where the write access
; is not through the original memory access, but trough a PHI node that was
; delicmed. This test covers the polybench 2mm and 3mm cases.
;
; CHECK: The matrix multiplication pattern was detected
;

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @kernel_2mm(i32 %ni, i32 %nj, i32 %nk, i32 %nl, double %alpha, double %beta, [1800 x double]* nocapture %tmp, [2200 x double]* nocapture readonly %A, [1800 x double]* nocapture readonly %B, [2400 x double]* nocapture readnone %C, [2400 x double]* nocapture readnone %D) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc25, %entry.split
  %indvars.iv50 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next51, %for.inc25 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc22, %for.body
  %indvars.iv46 = phi i64 [ 0, %for.body ], [ %indvars.iv.next47, %for.inc22 ]
  %arrayidx5 = getelementptr inbounds [1800 x double], [1800 x double]* %tmp, i64 %indvars.iv50, i64 %indvars.iv46
  store double 0.000000e+00, double* %arrayidx5, align 8, !tbaa !2
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body3
  %0 = phi double [ 0.000000e+00, %for.body3 ], [ %add, %for.body8 ]
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx12 = getelementptr inbounds [2200 x double], [2200 x double]* %A, i64 %indvars.iv50, i64 %indvars.iv
  %1 = load double, double* %arrayidx12, align 8, !tbaa !2
  %mul = fmul double %1, %alpha
  %arrayidx16 = getelementptr inbounds [1800 x double], [1800 x double]* %B, i64 %indvars.iv, i64 %indvars.iv46
  %2 = load double, double* %arrayidx16, align 8, !tbaa !2
  %mul17 = fmul double %mul, %2
  %add = fadd double %0, %mul17
  store double %add, double* %arrayidx5, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2200
  br i1 %exitcond, label %for.inc22, label %for.body8

for.inc22:                                        ; preds = %for.body8
  %indvars.iv.next47 = add nuw nsw i64 %indvars.iv46, 1
  %exitcond48 = icmp eq i64 %indvars.iv.next47, 1800
  br i1 %exitcond48, label %for.inc25, label %for.body3

for.inc25:                                        ; preds = %for.inc22
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %exitcond52 = icmp eq i64 %indvars.iv.next51, 1600
  br i1 %exitcond52, label %for.end27, label %for.body

for.end27:                                        ; preds = %for.inc25
  ret void
}

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vl,-avx512vpopcntdq,-clflushopt,-clwb,-clzero,-fma4,-lwp,-mwaitx,-pku,-prefetchwt1,-prfchw,-rdseed,-rtm,-sgx,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (trunk 309912) (llvm/trunk 309933)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
