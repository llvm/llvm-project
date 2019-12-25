; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-optree-normalize-phi=true -polly-optree -analyze < %s | FileCheck %s -match-full-lines

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define internal fastcc void @kernel_atax([2100 x double]* nocapture readonly %A, double* nocapture readonly %x, double* nocapture %y, double* nocapture %tmp) unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %y15 = bitcast double* %y to i8*
  call void @llvm.memset.p0i8.i64(i8* %y15, i8 0, i64 16800, i32 8, i1 false)
  br label %for.body3

for.body3:                                        ; preds = %for.inc40, %entry.split
  %indvars.iv8 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next9, %for.inc40 ]
  %arrayidx5 = getelementptr inbounds double, double* %tmp, i64 %indvars.iv8
  store double 0.000000e+00, double* %arrayidx5, align 8, !tbaa !6
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body3
  %0 = phi double [ 0.000000e+00, %for.body3 ], [ %add, %for.body8 ]
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx14 = getelementptr inbounds [2100 x double], [2100 x double]* %A, i64 %indvars.iv8, i64 %indvars.iv
  %1 = load double, double* %arrayidx14, align 8, !tbaa !6
  %arrayidx16 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %2 = load double, double* %arrayidx16, align 8, !tbaa !6
  %mul = fmul double %1, %2
  %add = fadd double %0, %mul
  store double %add, double* %arrayidx5, align 8, !tbaa !6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2
  br i1 %exitcond, label %for.end21, label %for.body8

for.end21:                                        ; preds = %for.body8
  br label %for.body24

for.body24:                                       ; preds = %for.body24.for.body24_crit_edge, %for.end21
  %3 = phi double [ %add, %for.end21 ], [ %.pre, %for.body24.for.body24_crit_edge ]
  %indvars.iv5 = phi i64 [ 0, %for.end21 ], [ %indvars.iv.next6, %for.body24.for.body24_crit_edge ]
  %arrayidx26 = getelementptr inbounds double, double* %y, i64 %indvars.iv5
  %4 = load double, double* %arrayidx26, align 8, !tbaa !6
  %arrayidx30 = getelementptr inbounds [2100 x double], [2100 x double]* %A, i64 %indvars.iv8, i64 %indvars.iv5
  %5 = load double, double* %arrayidx30, align 8, !tbaa !6
  %mul33 = fmul double %5, %3
  %add34 = fadd double %4, %mul33
  store double %add34, double* %arrayidx26, align 8, !tbaa !6
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %exitcond7 = icmp eq i64 %indvars.iv.next6, 2
  br i1 %exitcond7, label %for.inc40, label %for.body24.for.body24_crit_edge

for.body24.for.body24_crit_edge:                  ; preds = %for.body24
  %.pre = load double, double* %arrayidx5, align 8, !tbaa !6
  br label %for.body24

for.inc40:                                        ; preds = %for.body24
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %exitcond10 = icmp eq i64 %indvars.iv.next9, 2
  br i1 %exitcond10, label %for.end42, label %for.body3

for.end42:                                        ; preds = %for.inc40
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (trunk 312565) (llvm/trunk 312564)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}


; CHECK: Statistics {
; CHECK:     Operand trees forwarded: 2
; CHECK:     Statements with forwarded operand trees: 2
; CHECK: }

; CHECK-NEXT: After statements {
; CHECK-NEXT:     Stmt_for_body3
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body3[i0] -> MemRef_tmp[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body3[i0] -> MemRef1__phi[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   store double 0.000000e+00, double* %arrayidx5, align 8, !tbaa !2
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_for_body8
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body8[i0, i1] -> MemRef1__phi[] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body8[i0, i1] -> MemRef1__phi[] };
; CHECK-NEXT:            new: { Stmt_for_body8[i0, i1] -> MemRef_tmp[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body8[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body8[i0, i1] -> MemRef_x[i1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body8[i0, i1] -> MemRef_tmp[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body8[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %0 = phi double [ 0.000000e+00, %for.body3 ], [ %add, %for.body8 ]
; CHECK-NEXT:                   %1 = load double, double* %arrayidx14, align 8, !tbaa !2
; CHECK-NEXT:                   %2 = load double, double* %arrayidx16, align 8, !tbaa !2
; CHECK-NEXT:                   %mul = fmul double %1, %2
; CHECK-NEXT:                   %add = fadd double %0, %mul
; CHECK-NEXT:                   store double %add, double* %arrayidx5, align 8, !tbaa !2
; CHECK-NEXT:                   %exitcond = icmp eq i64 %indvars.iv.next, 2
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_for_end21
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_end21[i0] -> MemRef_add[] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_end21[i0] -> MemRef5__phi[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_for_body24
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body24[i0, i1] -> MemRef5__phi[] };
; CHECK-NEXT:            new: { Stmt_for_body24[i0, i1] -> MemRef_tmp[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body24[i0, i1] -> MemRef_y[i1] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body24[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body24[i0, i1] -> MemRef_y[i1] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %3 = phi double [ %add, %for.end21 ], [ %.pre, %for.body24.for.body24_crit_edge ]
; CHECK-NEXT:                   %4 = load double, double* %arrayidx26, align 8, !tbaa !2
; CHECK-NEXT:                   %5 = load double, double* %arrayidx30, align 8, !tbaa !2
; CHECK-NEXT:                   %mul33 = fmul double %5, %3
; CHECK-NEXT:                   %add34 = fadd double %4, %mul33
; CHECK-NEXT:                   store double %add34, double* %arrayidx26, align 8, !tbaa !2
; CHECK-NEXT:                   %exitcond7 = icmp eq i64 %indvars.iv.next6, 2
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_for_body24_for_body24_crit_edge
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body24_for_body24_crit_edge[i0, i1] -> MemRef5__phi[] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body24_for_body24_crit_edge[i0, i1] -> MemRef_tmp[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %.pre = load double, double* %arrayidx5, align 8, !tbaa !2
; CHECK-NEXT:             }
; CHECK-NEXT: }
