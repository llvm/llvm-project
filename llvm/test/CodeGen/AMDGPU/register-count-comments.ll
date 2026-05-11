; RUN: llc -mtriple=amdgcn -asm-verbose < %s | FileCheck -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn-unknown-amdhsa -asm-verbose -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #0

; SI-LABEL: {{^}}foo:
define amdgpu_kernel void @foo(ptr addrspace(1) noalias %out, ptr addrspace(1) %abase, ptr addrspace(1) %bbase) nounwind {
  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0);
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbcnt.lo)
  %aptr = getelementptr i32, ptr addrspace(1) %abase, i32 %tid
  %bptr = getelementptr i32, ptr addrspace(1) %bbase, i32 %tid
  %outptr = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  %a = load i32, ptr addrspace(1) %aptr, align 4
  %b = load i32, ptr addrspace(1) %bptr, align 4
  %result = add i32 %a, %b
  store i32 %result, ptr addrspace(1) %outptr, align 4
  ret void
}

; SI-LABEL: {{^}}one_vgpr_used:
define amdgpu_kernel void @one_vgpr_used(ptr addrspace(1) %out, i32 %x) nounwind {
  store i32 %x, ptr addrspace(1) %out, align 4
  ret void
}

; SI: .section	.AMDGPU.csdata
; SI: ; foo Kernel info:
; SI: ; TotalNumSgprs: {{[0-9]+}}
; SI: ; NumVgprs: {{[0-9]+}}
; SI: ; one_vgpr_used Kernel info:
; SI: ; TotalNumSgprs: {{[0-9]+}}
; SI: ; NumVgprs: 1
