; RUN: opt -mtriple=amdgcn-amd-amdhsa -basic-aa -load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

declare i64 @_Z12get_local_idj(i32)

declare i64 @_Z12get_group_idj(i32)

declare double @llvm.fmuladd.f64(double, double, double)

; CHECK-LABEL: @factorizedVsNonfactorizedAccess(
; CHECK: load <2 x float>
; CHECK: store <2 x float>
define amdgpu_kernel void @factorizedVsNonfactorizedAccess(float addrspace(1)* nocapture %c) {
entry:
  %call = tail call i64 @_Z12get_local_idj(i32 0)
  %call1 = tail call i64 @_Z12get_group_idj(i32 0)
  %div = lshr i64 %call, 4
  %div2 = lshr i64 %call1, 3
  %mul = shl i64 %div2, 7
  %rem = shl i64 %call, 3
  %mul3 = and i64 %rem, 120
  %add = or i64 %mul, %mul3
  %rem4 = shl i64 %call1, 7
  %mul5 = and i64 %rem4, 896
  %mul6 = shl nuw nsw i64 %div, 3
  %add7 = add nuw i64 %mul5, %mul6
  %mul9 = shl i64 %add7, 10
  %add10 = add i64 %mul9, %add
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %c, i64 %add10
  %load1 = load float, float addrspace(1)* %arrayidx, align 4
  %conv = fpext float %load1 to double
  %mul11 = fmul double %conv, 0x3FEAB481D8F35506
  %conv12 = fptrunc double %mul11 to float
  %conv18 = fpext float %conv12 to double
  %storeval1 = tail call double @llvm.fmuladd.f64(double 0x3FF4FFAFBBEC946A, double 0.000000e+00, double %conv18)
  %cstoreval1 = fptrunc double %storeval1 to float
  store float %cstoreval1, float addrspace(1)* %arrayidx, align 4

  %add23 = or i64 %add10, 1
  %arrayidx24 = getelementptr inbounds float, float addrspace(1)* %c, i64 %add23
  %load2 = load float, float addrspace(1)* %arrayidx24, align 4
  %conv25 = fpext float %load2 to double
  %mul26 = fmul double %conv25, 0x3FEAB481D8F35506
  %conv27 = fptrunc double %mul26 to float
  %conv34 = fpext float %conv27 to double
  %storeval2 = tail call double @llvm.fmuladd.f64(double 0x3FF4FFAFBBEC946A, double 0.000000e+00, double %conv34)
  %cstoreval2 = fptrunc double %storeval2 to float
  store float %cstoreval2, float addrspace(1)* %arrayidx24, align 4
  ret void
}
