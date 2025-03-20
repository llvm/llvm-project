; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn < %s -debug-only=igrouplp 2>&1 | FileCheck --check-prefixes=GCN %s




; Inverted 1008: 001111110000 
; GCN: After Inverting, SchedGroup Mask: 1008
define amdgpu_kernel void @invert1() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 4092: 111111111100 
; GCN:       After Inverting, SchedGroup Mask: 4092
define amdgpu_kernel void @invert2() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 2) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 4090: 111111111010
; GCN:       After Inverting, SchedGroup Mask: 4090
define amdgpu_kernel void @invert4() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 4) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 4086: 111111110110
; GCN:       After Inverting, SchedGroup Mask: 4086
define amdgpu_kernel void @invert8() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 8) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 3983: 111110001111
; GCN:       After Inverting, SchedGroup Mask: 3983
define amdgpu_kernel void @invert16() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 16) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 4047: 111111001111
; GCN:       After Inverting, SchedGroup Mask: 4047
define amdgpu_kernel void @invert32() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 32) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 4015: 111110101111
; GCN:       After Inverting, SchedGroup Mask: 4015
define amdgpu_kernel void @invert64() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 64) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 3199: 110001111111
; GCN:       After Inverting, SchedGroup Mask: 3199
define amdgpu_kernel void @invert128() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 128) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 3711: 111001111111
; GCN:       After Inverting, SchedGroup Mask: 3711
define amdgpu_kernel void @invert256() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 256) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 3455: 110101111111
; GCN:       After Inverting, SchedGroup Mask: 3455
define amdgpu_kernel void @invert512() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 512) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 3070: 101111111110
; GCN:       After Inverting, SchedGroup Mask: 3070
define amdgpu_kernel void @invert1024() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1024) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 2046: 011111111110
; GCN:       After Inverting, SchedGroup Mask: 2046
define amdgpu_kernel void @invert2048() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 2048) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}


declare void @llvm.amdgcn.sched.barrier(i32) #1
declare void @llvm.amdcn.s.nop(i16) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
