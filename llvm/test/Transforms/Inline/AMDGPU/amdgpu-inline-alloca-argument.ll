; RUN: opt -mtriple=amdgcn--amdhsa -S -inline -inline-threshold=0 < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes=inline -inline-threshold=0 < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define void @use_flat_ptr_arg(float* nocapture %p) {
entry:
  %tmp1 = load float, float* %p, align 4
  %div = fdiv float 1.000000e+00, %tmp1
  %add0 = fadd float %div, 1.0
  %add1 = fadd float %add0, 1.0
  %add2 = fadd float %add1, 1.0
  %add3 = fadd float %add2, 1.0
  %add4 = fadd float %add3, 1.0
  %add5 = fadd float %add4, 1.0
  %add6 = fadd float %add5, 1.0
  %add7 = fadd float %add6, 1.0
  %add8 = fadd float %add7, 1.0
  %add9 = fadd float %add8, 1.0
  %add10 = fadd float %add9, 1.0
  store float %add10, float* %p, align 4
  ret void
}

define void @use_private_ptr_arg(float addrspace(5)* nocapture %p) {
entry:
  %tmp1 = load float, float addrspace(5)* %p, align 4
  %div = fdiv float 1.000000e+00, %tmp1
  %add0 = fadd float %div, 1.0
  %add1 = fadd float %add0, 1.0
  %add2 = fadd float %add1, 1.0
  %add3 = fadd float %add2, 1.0
  %add4 = fadd float %add3, 1.0
  %add5 = fadd float %add4, 1.0
  %add6 = fadd float %add5, 1.0
  %add7 = fadd float %add6, 1.0
  %add8 = fadd float %add7, 1.0
  %add9 = fadd float %add8, 1.0
  %add10 = fadd float %add9, 1.0
  store float %add10, float addrspace(5)* %p, align 4
  ret void
}

; Test that the inline threshold is boosted if called with an
; addrspacecasted' alloca.
; CHECK-LABEL: @test_inliner_flat_ptr(
; CHECK: call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NOT: call
; CHECK-NOT: call
define amdgpu_kernel void @test_inliner_flat_ptr(float addrspace(1)* nocapture %a, i32 %n) {
entry:
  %pvt_arr = alloca [64 x float], align 4, addrspace(5)
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %a, i32 %tid
  %tmp2 = load float, float addrspace(1)* %arrayidx, align 4
  %add = add i32 %tid, 1
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %a, i32 %add
  %tmp5 = load float, float addrspace(1)* %arrayidx2, align 4
  %or = or i32 %tid, %n
  %arrayidx5 = getelementptr inbounds [64 x float], [64 x float] addrspace(5)* %pvt_arr, i32 0, i32 %or
  %arrayidx7 = getelementptr inbounds [64 x float], [64 x float] addrspace(5)* %pvt_arr, i32 0, i32 %or
  %to.flat = addrspacecast float addrspace(5)* %arrayidx7 to float*
  call void @use_private_ptr_arg(float addrspace(5)* %arrayidx7)
  call void @use_flat_ptr_arg(float* %to.flat)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { noinline }
attributes #1 = { nounwind readnone }
