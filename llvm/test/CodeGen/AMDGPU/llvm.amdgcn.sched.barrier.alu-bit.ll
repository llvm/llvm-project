; RUN: llc -mtriple=amdgcn -debug-only=igrouplp < %s 2>&1| FileCheck -check-prefix=GCN %s

define protected amdgpu_kernel void @sched_barrier_m0(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 0 (no bits set)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Applying IGroupLPDAGMutation...
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 0) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m1(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 1 (ALU Bit, implies all *-ALU bits)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 1
; GCN-NEXT: After Inverting, SchedGroup Mask: 1008
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 1) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m2(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 2 (VALU Bit)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 2
; GCN-NEXT: After Inverting, SchedGroup Mask: 2044
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 2) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m4(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 4 (SALU Bit)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 4
; GCN-NEXT: After Inverting, SchedGroup Mask: 2042
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 4) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m8(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 8 (MFMA Bit)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 8
; GCN-NEXT: After Inverting, SchedGroup Mask: 2038
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 8) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m1024(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 1024 (TRANS Bit)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 1024
; GCN-NEXT: After Inverting, SchedGroup Mask: 1022
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 1024) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m3(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 3 (ALU + VALU Bits)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 3
; GCN-NEXT: After Inverting, SchedGroup Mask: 2044
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 3) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m5(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 5 (ALU + SALU Bits)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 5
; GCN-NEXT: After Inverting, SchedGroup Mask: 2042
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 5) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m7(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 7 (ALU + VALU + SALU Bits)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 7
; GCN-NEXT: After Inverting, SchedGroup Mask: 2040
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 7) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

define protected amdgpu_kernel void @sched_barrier_m15(ptr addrspace(3) noalias %ind, ptr addrspace(3) noalias %outd) #0 {
;
; Set mask to 15 (ALU + VALU + SALU + MFMA Bits)
;
; GCN: Applying IGroupLPDAGMutation...
; GCN-NEXT: Building SchedGroup for SchedBarrier with Mask: 15
; GCN-NEXT: After Inverting, SchedGroup Mask: 2032
entry:
  ; we need salu, valu, mfma, trans instructions here.
  %arrayidx = getelementptr inbounds float, ptr addrspace(3) %ind, i64 0
  %1 = load float, ptr addrspace(3) %arrayidx, align 4
  call void @llvm.amdgcn.sched.barrier(i32 15) #1
  %add = fadd contract float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %outd, i64 0
  store float %add, ptr addrspace(3) %arrayidx3, align 4
  ret void
}

declare void @llvm.amdgcn.sched.barrier(i32) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
