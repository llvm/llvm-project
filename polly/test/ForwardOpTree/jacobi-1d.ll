; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb -polly-optree-normalize-phi=true '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define internal fastcc void @kernel_jacobi_1d(ptr noalias nocapture %A, ptr noalias nocapture %B) unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %arrayidx6.phi.trans.insert = getelementptr inbounds double, ptr %A, i64 1
  %arrayidx21.phi.trans.insert = getelementptr inbounds double, ptr %B, i64 1
  br label %for.body

for.body:                                         ; preds = %for.inc33, %entry.split
  %t.03 = phi i32 [ 0, %entry.split ], [ %inc34, %for.inc33 ]
  %.pre = load double, ptr %A, align 8, !tbaa !6
  %.pre10 = load double, ptr %arrayidx6.phi.trans.insert, align 8, !tbaa !6
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %0 = phi double [ %.pre10, %for.body ], [ %2, %for.body3 ]
  %1 = phi double [ %.pre, %for.body ], [ %0, %for.body3 ]
  %indvars.iv = phi i64 [ 1, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %add = fadd double %1, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx9 = getelementptr inbounds double, ptr %A, i64 %indvars.iv.next
  %2 = load double, ptr %arrayidx9, align 8, !tbaa !6
  %add10 = fadd double %add, %2
  %mul = fmul double %add10, 3.333300e-01
  %arrayidx12 = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  store double %mul, ptr %arrayidx12, align 8, !tbaa !6
  %exitcond = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond, label %for.end, label %for.body3

for.end:                                          ; preds = %for.body3
  %.pre11 = load double, ptr %B, align 8, !tbaa !6
  %.pre12 = load double, ptr %arrayidx21.phi.trans.insert, align 8, !tbaa !6
  br label %for.inc33

for.inc33:                                        ; preds = %for.body16
  %inc34 = add nuw nsw i32 %t.03, 1
  %exitcond9 = icmp eq i32 %inc34, 2
  br i1 %exitcond9, label %for.end35, label %for.body

for.end35:                                        ; preds = %for.inc33
  ret void
}

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0  (llvm/trunk 312874)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}


; CHECK: Statistics {
; CHECK:     Operand trees forwarded: 2
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK-NEXT: After statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body[i0] -> MemRef_A[1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body[i0] -> MemRef1__phi[] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body[i0] -> MemRef2__phi[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %.pre = load double, ptr %A, align 8, !tbaa !2
; CHECK-NEXT:                   %.pre10 = load double, ptr %arrayidx6.phi.trans.insert, align 8, !tbaa !2
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_for_body3
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body3[i0, i1] -> MemRef1__phi[] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body3[i0, i1] -> MemRef1__phi[] };
; CHECK-NEXT:            new: { Stmt_for_body3[i0, i1] -> MemRef_A[1 + i1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body3[i0, i1] -> MemRef2__phi[] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body3[i0, i1] -> MemRef2__phi[] };
; CHECK-NEXT:            new: { Stmt_for_body3[i0, i1] -> MemRef_A[i1] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body3[i0, i1] -> MemRef_A[2 + i1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_for_body3[i0, i1] -> MemRef_B[1 + i1] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %0 = phi double [ %.pre10, %for.body ], [ %2, %for.body3 ]
; CHECK-NEXT:                   %1 = phi double [ %.pre, %for.body ], [ %0, %for.body3 ]
; CHECK-NEXT:                   %add = fadd double %1, %0
; CHECK-NEXT:                   %2 = load double, ptr %arrayidx9, align 8, !tbaa !2
; CHECK-NEXT:                   %add10 = fadd double %add, %2
; CHECK-NEXT:                   %mul = fmul double %add10, 3.333300e-01
; CHECK-NEXT:                   store double %mul, ptr %arrayidx12, align 8, !tbaa !2
; CHECK-NEXT:                   %exitcond = icmp eq i64 %indvars.iv.next, 3
; CHECK-NEXT:             }
; CHECK-NEXT: }
