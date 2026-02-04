; RUN: opt -S -mtriple=amdgcn-amd-amdhsa --amdgpu-annotate-uniform < %s | FileCheck -check-prefix=OPT %s

; "load vaddr" depends on the store, so we should not mark vaddr as amdgpu.noclobber.

; OPT-LABEL: @store_clobbers_load(
; OPT: %zero = load <4 x i32>, ptr addrspace(1) %input, align 16{{$}}
define amdgpu_kernel void @store_clobbers_load(ptr addrspace(1) %input,  ptr addrspace(1) %out, i32 %index) {
entry:
  store i32 0, ptr addrspace(1) %input
  %zero = load <4 x i32>, ptr addrspace(1) %input, align 16
  %one = insertelement <4 x i32> %zero, i32 1, i32 1
  %two = insertelement <4 x i32> %one, i32 2, i32 2
  %three = insertelement <4 x i32> %two, i32 3, i32 3
  store <4 x i32> %three, ptr addrspace(1) %input, align 16
  %rslt = extractelement <4 x i32> %three, i32 %index
  store i32 %rslt, ptr addrspace(1) %out, align 4
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x()
@lds0 = addrspace(3) global [512 x i32] poison, align 4

; To check that %arrayidx0 is not marked as amdgpu.noclobber.

; OPT-LABEL: @atomicrmw_clobbers_load(
; OPT:       %arrayidx0 = getelementptr inbounds [512 x i32], ptr addrspace(3) @lds0, i32 0, i32 %idx.0
; OPT-NEXT:  %val = atomicrmw xchg ptr addrspace(3) %arrayidx0, i32 3 seq_cst

define amdgpu_kernel void @atomicrmw_clobbers_load(ptr addrspace(1) %out0, ptr addrspace(1) %out1) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], ptr addrspace(3) @lds0, i32 0, i32 %idx.0
  %val = atomicrmw xchg ptr addrspace(3) %arrayidx0, i32 3 seq_cst
  %load = load i32, ptr addrspace(3) %arrayidx0, align 4
  store i32 %val, ptr addrspace(1) %out0, align 4
  store i32 %load, ptr addrspace(1) %out1, align 4
  ret void
}
