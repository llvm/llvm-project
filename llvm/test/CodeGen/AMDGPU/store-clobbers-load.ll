; RUN: opt -S --amdgpu-annotate-uniform < %s | FileCheck -check-prefix=OPT %s
target datalayout = "A5"

; "load vaddr" depends on the store, so we should not mark vaddr as amdgpu.noclobber.

; OPT-LABEL: @store_clobbers_load(
; OPT:      %vaddr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %input, i64 0, !amdgpu.uniform !0
; OPT-NEXT: %zero = load <4 x i32>, <4 x i32> addrspace(1)* %vaddr, align 16
define amdgpu_kernel void @store_clobbers_load( < 4 x i32> addrspace(1)* %input,  i32 addrspace(1)* %out, i32 %index) {
entry:
  %addr0 = bitcast <4 x i32> addrspace(1)* %input to i32 addrspace(1)*
  store i32 0, i32 addrspace(1)* %addr0
  %vaddr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %input, i64 0
  %zero = load <4 x i32>, <4 x i32> addrspace(1)* %vaddr, align 16
  %one = insertelement <4 x i32> %zero, i32 1, i32 1
  %two = insertelement <4 x i32> %one, i32 2, i32 2
  %three = insertelement <4 x i32> %two, i32 3, i32 3
  store <4 x i32> %three, <4 x i32> addrspace(1)* %input, align 16
  %rslt = extractelement <4 x i32> %three, i32 %index
  store i32 %rslt, i32 addrspace(1)* %out, align 4
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x()
@lds0 = addrspace(3) global [512 x i32] undef, align 4

; To check that %arrayidx0 is not marked as amdgpu.noclobber.

; OPT-LABEL: @atomicrmw_clobbers_load(
; OPT:       %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds0, i32 0, i32 %idx.0, !amdgpu.uniform !0
; OPT-NEXT:  %val = atomicrmw xchg i32 addrspace(3)* %arrayidx0, i32 3 seq_cst

define amdgpu_kernel void @atomicrmw_clobbers_load(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1) {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val = atomicrmw xchg i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  %load = load i32, i32 addrspace(3)* %arrayidx0, align 4
  store i32 %val, i32 addrspace(1)* %out0, align 4
  store i32 %load, i32 addrspace(1)* %out1, align 4
  ret void
}
