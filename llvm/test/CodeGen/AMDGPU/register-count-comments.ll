; RUN: llc -march=amdgcn -verify-machineinstrs -asm-verbose < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -verify-machineinstrs -asm-verbose < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.SI.tid() nounwind readnone

; SI-LABEL: {{^}}foo:
; SI: .section	.AMDGPU.csdata
; SI: ; Kernel info:
; SI: ; NumSgprs: {{[0-9]+}}
; SI: ; NumVgprs: {{[0-9]+}}
define void @foo(i32 addrspace(1)* noalias %out, i32 addrspace(1)* %abase, i32 addrspace(1)* %bbase) nounwind {
  %tid = call i32 @llvm.SI.tid() nounwind readnone
  %aptr = getelementptr i32, i32 addrspace(1)* %abase, i32 %tid
  %bptr = getelementptr i32, i32 addrspace(1)* %bbase, i32 %tid
  %outptr = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %outptr, align 4
  ret void
}

; SI-LABEL: {{^}}one_vgpr_used:
; SI: NumVgprs: 1
define void @one_vgpr_used(i32 addrspace(1)* %out, i32 %x) nounwind {
  store i32 %x, i32 addrspace(1)* %out, align 4
  ret void
}
