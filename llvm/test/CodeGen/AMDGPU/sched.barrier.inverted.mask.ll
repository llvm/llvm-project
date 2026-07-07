; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn -mcpu=gfx600 < %s -debug-only=igrouplp 2>&1 | FileCheck --check-prefixes=GCN %s

; Barriers are processed in reverse order
; GCN: After Inverting, SchedGroup Mask: 1903
; GCN: After Inverting, SchedGroup Mask: 3070
; GCN: After Inverting, SchedGroup Mask: 3455
; GCN: After Inverting, SchedGroup Mask: 3711
; GCN: After Inverting, SchedGroup Mask: 1151
; GCN: After Inverting, SchedGroup Mask: 4015
; GCN: After Inverting, SchedGroup Mask: 4047
; GCN: After Inverting, SchedGroup Mask: 1807
; GCN: After Inverting, SchedGroup Mask: 4086
; GCN: After Inverting, SchedGroup Mask: 4090
; GCN: After Inverting, SchedGroup Mask: 4092
; GCN: After Inverting, SchedGroup Mask: 3056
define amdgpu_kernel void @invert() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1) #1
  call void @llvm.amdgcn.sched.barrier(i32 2) #1
  call void @llvm.amdgcn.sched.barrier(i32 4) #1
  call void @llvm.amdgcn.sched.barrier(i32 8) #1
  call void @llvm.amdgcn.sched.barrier(i32 16) #1
  call void @llvm.amdgcn.sched.barrier(i32 32) #1
  call void @llvm.amdgcn.sched.barrier(i32 64) #1
  call void @llvm.amdgcn.sched.barrier(i32 128) #1
  call void @llvm.amdgcn.sched.barrier(i32 256) #1
  call void @llvm.amdgcn.sched.barrier(i32 512) #1
  call void @llvm.amdgcn.sched.barrier(i32 1024) #1
  call void @llvm.amdgcn.sched.barrier(i32 2048) #1
  call void @llvm.amdgcn.s.nop(i16 0) #1
  ret void
}

declare void @llvm.amdgcn.sched.barrier(i32) #1
declare void @llvm.amdgcn.s.nop(i16) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
