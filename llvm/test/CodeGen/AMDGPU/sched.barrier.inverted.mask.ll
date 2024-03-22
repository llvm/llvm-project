; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn < %s -debug-only=igrouplp 2>&1 | FileCheck --check-prefixes=GCN %s




; Inverted 1008: 01111110000 
; GCN: After Inverting, SchedGroup Mask: 1008
define amdgpu_kernel void @invert1() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 2044: 11111111100 
; GCN:       After Inverting, SchedGroup Mask: 2044
define amdgpu_kernel void @invert2() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 2) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 2042: 11111111010
; GCN:       After Inverting, SchedGroup Mask: 2042
define amdgpu_kernel void @invert4() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 4) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 2038: 11111110110
; GCN:       After Inverting, SchedGroup Mask: 2038
define amdgpu_kernel void @invert8() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 8) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1935: 11110001111
; GCN:       After Inverting, SchedGroup Mask: 1935
define amdgpu_kernel void @invert16() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 16) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1999: 11111001111
; GCN:       After Inverting, SchedGroup Mask: 1999
define amdgpu_kernel void @invert32() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 32) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1967: 11110101111
; GCN:       After Inverting, SchedGroup Mask: 1967
define amdgpu_kernel void @invert64() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 64) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1151: 10001111111
; GCN:       After Inverting, SchedGroup Mask: 1151
define amdgpu_kernel void @invert128() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 128) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1663: 11001111111
; GCN:       After Inverting, SchedGroup Mask: 1663
define amdgpu_kernel void @invert256() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 256) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1407: 10101111111
; GCN:       After Inverting, SchedGroup Mask: 1407
define amdgpu_kernel void @invert512() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 512) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

; Inverted 1022: 01111111110
; GCN:       After Inverting, SchedGroup Mask: 1022
define amdgpu_kernel void @invert1024() nounwind {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 1024) convergent nounwind
  call void @llvm.amdcn.s.nop(i16 0) convergent nounwind
  ret void
}

declare void @llvm.amdgcn.sched.barrier(i32) convergent nounwind
declare void @llvm.amdcn.s.nop(i16) convergent nounwind
