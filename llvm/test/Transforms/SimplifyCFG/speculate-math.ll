; RUN: opt -S -simplifycfg < %s | FileCheck %s --check-prefix=EXPENSIVE --check-prefix=ALL
; RUN: opt -S -simplifycfg -speculate-one-expensive-inst=false < %s | FileCheck %s --check-prefix=CHEAP --check-prefix=ALL

declare float @llvm.sqrt.f32(float) nounwind readonly
declare float @llvm.fma.f32(float, float, float) nounwind readonly
declare float @llvm.fmuladd.f32(float, float, float) nounwind readonly
declare float @llvm.fabs.f32(float) nounwind readonly
declare float @llvm.minnum.f32(float, float) nounwind readonly
declare float @llvm.maxnum.f32(float, float) nounwind readonly
declare float @llvm.minimum.f32(float, float) nounwind readonly
declare float @llvm.maximum.f32(float, float) nounwind readonly

; ALL-LABEL: @fdiv_test(
; EXPENSIVE: select i1 %cmp, double %div, double 0.0
; CHEAP-NOT: select

define double @fdiv_test(double %a, double %b) {
entry:
  %cmp = fcmp ogt double %a, 0.0
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:
  %div = fdiv double %b, %a
  br label %cond.end

cond.end:
  %cond = phi double [ %div, %cond.true ], [ 0.0, %entry ]
  ret double %cond
}

; ALL-LABEL: @sqrt_test(
; ALL: select
define void @sqrt_test(float addrspace(1)* noalias nocapture %out, float %a) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_sqrt.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.sqrt.f32(float %a) nounwind readnone
  br label %test_sqrt.exit

test_sqrt.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @fabs_test(
; ALL: select
define void @fabs_test(float addrspace(1)* noalias nocapture %out, float %a) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_fabs.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.fabs.f32(float %a) nounwind readnone
  br label %test_fabs.exit

test_fabs.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @fma_test(
; ALL: select
define void @fma_test(float addrspace(1)* noalias nocapture %out, float %a, float %b, float %c) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_fma.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c) nounwind readnone
  br label %test_fma.exit

test_fma.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @fmuladd_test(
; ALL: select
define void @fmuladd_test(float addrspace(1)* noalias nocapture %out, float %a, float %b, float %c) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_fmuladd.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.fmuladd.f32(float %a, float %b, float %c) nounwind readnone
  br label %test_fmuladd.exit

test_fmuladd.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @minnum_test(
; ALL: select
define void @minnum_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_minnum.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.minnum.f32(float %a, float %b) nounwind readnone
  br label %test_minnum.exit

test_minnum.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @maxnum_test(
; ALL: select
define void @maxnum_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_maxnum.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.maxnum.f32(float %a, float %b) nounwind readnone
  br label %test_maxnum.exit

test_maxnum.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @minimum_test(
; ALL: select
define void @minimum_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_minimum.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.minimum.f32(float %a, float %b) nounwind readnone
  br label %test_minimum.exit

test_minimum.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}

; ALL-LABEL: @maximum_test(
; ALL: select
define void @maximum_test(float addrspace(1)* noalias nocapture %out, float %a, float %b) nounwind {
entry:
  %cmp.i = fcmp olt float %a, 0.000000e+00
  br i1 %cmp.i, label %test_maximum.exit, label %cond.else.i

cond.else.i:                                      ; preds = %entry
  %0 = tail call float @llvm.maximum.f32(float %a, float %b) nounwind readnone
  br label %test_maximum.exit

test_maximum.exit:                                   ; preds = %cond.else.i, %entry
  %cond.i = phi float [ %0, %cond.else.i ], [ 0x7FF8000000000000, %entry ]
  store float %cond.i, float addrspace(1)* %out, align 4
  ret void
}
