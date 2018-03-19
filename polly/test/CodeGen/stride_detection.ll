; RUN: opt %loadPolly -polly-opt-isl -polly-vectorizer=polly -polly-codegen < %s -S | FileCheck %s

; #pragma known-parallel
;   for (int c0 = 0; c0 <= 31; c0 += 1)
;     for (int c1 = 0; c1 <= floord(nk - 1, 32); c1 += 1)
;       for (int c2 = 0; c2 <= 7; c2 += 1)
;         for (int c3 = 0; c3 <= min(31, nk - 32 * c1 - 1); c3 += 1)
;           #pragma simd
;           for (int c4 = 0; c4 <= 3; c4 += 1)
;             Stmt_for_body_3(32 * c0 + 4 * c2 + c4, 32 * c1 + c3);

; CHECK: polly.stmt.for.body.3:                            ; preds = %polly.loop_header18
; CHECK:   %_p_splat_one = load <1 x double>, <1 x double>* %_p_vec_p, align 8, !alias.scope !1, !noalias !3, !llvm.mem.parallel_loop_access !0
; CHECK:   %_p_vec_full = load <4 x double>, <4 x double>* %vector_ptr, align 8, !alias.scope !4, !noalias !5, !llvm.mem.parallel_loop_access !0
; CHECK:   extractelement <4 x double> %addp_vec, i32 0
; CHECK:   extractelement <4 x double> %addp_vec, i32 1
; CHECK:   extractelement <4 x double> %addp_vec, i32 2
; CHECK:   extractelement <4 x double> %addp_vec, i32 3
; CHECK:   store <4 x double> %addp_vec, <4 x double>* {{.*}}, align 8, !alias.scope !4, !noalias !5, !llvm.mem.parallel_loop_access !0

define void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, [1024 x double]* %C, [1024 x double]* %A) #0 {
entry:
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %entry, %for.inc.10
  %indvars.iv16 = phi i64 [ 0, %entry ], [ %indvars.iv.next17, %for.inc.10 ]
  %cmp2.13 = icmp sgt i32 %nk, 0
  br i1 %cmp2.13, label %for.body.3.lr.ph, label %for.inc.10

for.body.3.lr.ph:                                 ; preds = %for.cond.1.preheader
  br label %for.body.3

for.body.3:                                       ; preds = %for.body.3.lr.ph, %for.body.3
  %indvars.iv = phi i64 [ 0, %for.body.3.lr.ph ], [ %indvars.iv.next, %for.body.3 ]
  %arrayidx5 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 0, i64 %indvars.iv
  %0 = load double, double* %arrayidx5, align 8
  %arrayidx9 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 0, i64 %indvars.iv16
  %1 = load double, double* %arrayidx9, align 8
  %add = fadd double %0, %1
  store double %add, double* %arrayidx9, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %nk
  br i1 %exitcond, label %for.body.3, label %for.cond.1.for.inc.10_crit_edge

for.cond.1.for.inc.10_crit_edge:                  ; preds = %for.body.3
  br label %for.inc.10

for.inc.10:                                       ; preds = %for.cond.1.for.inc.10_crit_edge, %for.cond.1.preheader
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1
  %exitcond18 = icmp ne i64 %indvars.iv.next17, 1024
  br i1 %exitcond18, label %for.cond.1.preheader, label %for.end.12

for.end.12:                                       ; preds = %for.inc.10
  ret void
}
