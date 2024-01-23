; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn < %s -debug-only=igrouplp 2>&1 | FileCheck --check-prefixes=GCN %s




; Inverted 1008: 01111110000 
; GCN: After Inverting, SchedGroup Mask: 1008
define amdgpu_kernel void @invert1() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 2044: 11111111100 
; GCN:       After Inverting, SchedGroup Mask: 2044
define amdgpu_kernel void @invert2() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 2) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 2042: 11111111010
; GCN:       After Inverting, SchedGroup Mask: 2042
define amdgpu_kernel void @invert4() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 4) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 2038: 11111110110
; GCN:       After Inverting, SchedGroup Mask: 2038
define amdgpu_kernel void @invert8() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 8) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1935: 11110001111
; GCN:       After Inverting, SchedGroup Mask: 1935
define amdgpu_kernel void @invert16() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 16) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1999: 11111001111
; GCN:       After Inverting, SchedGroup Mask: 1999
define amdgpu_kernel void @invert32() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 32) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1967: 11110101111
; GCN:       After Inverting, SchedGroup Mask: 1967
define amdgpu_kernel void @invert64() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 64) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1151: 10001111111
; GCN:       After Inverting, SchedGroup Mask: 1151
define amdgpu_kernel void @invert128() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 128) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1663: 11001111111
; GCN:       After Inverting, SchedGroup Mask: 1663
define amdgpu_kernel void @invert256() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 256) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1407: 10101111111
; GCN:       After Inverting, SchedGroup Mask: 1407
define amdgpu_kernel void @invert512() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 512) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

; Inverted 1022: 01111111110
; GCN:       After Inverting, SchedGroup Mask: 1022
define amdgpu_kernel void @invert1024() #0 {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1024) #1
  call void @llvm.amdcn.s.nop(i16 0) #1
  ret void
}

declare void @llvm.amdgcn.sched.barrier(i32) #1
declare void @llvm.amdcn.s.nop(i16) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
