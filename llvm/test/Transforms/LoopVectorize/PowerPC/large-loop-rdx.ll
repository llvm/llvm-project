; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s

; CHECK: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: =
; CHECK-NOT: fadd
; CHECK-SAME: >

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-ibm-linux-gnu"

define void @QLA_F3_r_veq_norm2_V(float* noalias nocapture %r, [3 x { float, float }]* noalias nocapture readonly %a, i32 signext %n) #0 {
entry:
  %cmp24 = icmp sgt i32 %n, 0
  br i1 %cmp24, label %for.cond1.preheader.preheader, label %for.end13

for.cond1.preheader.preheader:                    ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.cond1.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.cond1.preheader ], [ 0, %for.cond1.preheader.preheader ]
  %sum.026 = phi double [ %add10.2, %for.cond1.preheader ], [ 0.000000e+00, %for.cond1.preheader.preheader ]
  %arrayidx5.realp = getelementptr inbounds [3 x { float, float }], [3 x { float, float }]* %a, i64 %indvars.iv, i64 0, i32 0
  %arrayidx5.real = load float, float* %arrayidx5.realp, align 8
  %arrayidx5.imagp = getelementptr inbounds [3 x { float, float }], [3 x { float, float }]* %a, i64 %indvars.iv, i64 0, i32 1
  %arrayidx5.imag = load float, float* %arrayidx5.imagp, align 8
  %mul = fmul fast float %arrayidx5.real, %arrayidx5.real
  %mul9 = fmul fast float %arrayidx5.imag, %arrayidx5.imag
  %add = fadd fast float %mul9, %mul
  %conv = fpext float %add to double
  %add10 = fadd fast double %conv, %sum.026
  %arrayidx5.realp.1 = getelementptr inbounds [3 x { float, float }], [3 x { float, float }]* %a, i64 %indvars.iv, i64 1, i32 0
  %arrayidx5.real.1 = load float, float* %arrayidx5.realp.1, align 8
  %arrayidx5.imagp.1 = getelementptr inbounds [3 x { float, float }], [3 x { float, float }]* %a, i64 %indvars.iv, i64 1, i32 1
  %arrayidx5.imag.1 = load float, float* %arrayidx5.imagp.1, align 8
  %mul.1 = fmul fast float %arrayidx5.real.1, %arrayidx5.real.1
  %mul9.1 = fmul fast float %arrayidx5.imag.1, %arrayidx5.imag.1
  %add.1 = fadd fast float %mul9.1, %mul.1
  %conv.1 = fpext float %add.1 to double
  %add10.1 = fadd fast double %conv.1, %add10
  %arrayidx5.realp.2 = getelementptr inbounds [3 x { float, float }], [3 x { float, float }]* %a, i64 %indvars.iv, i64 2, i32 0
  %arrayidx5.real.2 = load float, float* %arrayidx5.realp.2, align 8
  %arrayidx5.imagp.2 = getelementptr inbounds [3 x { float, float }], [3 x { float, float }]* %a, i64 %indvars.iv, i64 2, i32 1
  %arrayidx5.imag.2 = load float, float* %arrayidx5.imagp.2, align 8
  %mul.2 = fmul fast float %arrayidx5.real.2, %arrayidx5.real.2
  %mul9.2 = fmul fast float %arrayidx5.imag.2, %arrayidx5.imag.2
  %add.2 = fadd fast float %mul9.2, %mul.2
  %conv.2 = fpext float %add.2 to double
  %add10.2 = fadd fast double %conv.2, %add10.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.for.end13_crit_edge, label %for.cond1.preheader

for.cond.for.end13_crit_edge:                     ; preds = %for.cond1.preheader
  %add10.2.lcssa = phi double [ %add10.2, %for.cond1.preheader ]
  %phitmp = fptrunc double %add10.2.lcssa to float
  br label %for.end13

for.end13:                                        ; preds = %for.cond.for.end13_crit_edge, %entry
  %sum.0.lcssa = phi float [ %phitmp, %for.cond.for.end13_crit_edge ], [ 0.000000e+00, %entry ]
  store float %sum.0.lcssa, float* %r, align 4
  ret void
}

