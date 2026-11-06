; RUN: opt -S -mtriple=nvptx64 --mcpu=sm_86 -passes=loop-unroll -unroll-threshold=65 -unroll-max-iteration-count-to-analyze=16 < %s | FileCheck %s

; CHECK-LABEL: @test_asm_call_cost(
; CHECK-NOT: br i1
define void @test_asm_call_cost(ptr addrspace(1) noalias nocapture %in, ptr addrspace(1) noalias nocapture %out) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi float [ %fmul, %for.body ], [ 0.0, %entry ]
  %arrayidx.in = getelementptr inbounds float, ptr addrspace(1) %in, i32 %indvars.iv
  %arrayidx.out = getelementptr inbounds float, ptr addrspace(1) %out, i32 %indvars.iv
  %load = load float, ptr addrspace(1) %arrayidx.in
  %call = call float asm "add.f32 $0, $1, $2;", "=f,f,f"(float %load, float 1.0)
  %fmul = fmul float %call, %sum.02
  store float %fmul, ptr addrspace(1) %arrayidx.out
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
