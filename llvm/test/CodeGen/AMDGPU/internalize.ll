; RUN: opt -O1 -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=ALL -check-prefix=OPT %s
; RUN: opt -O0 -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=ALL -check-prefix=OPTNONE %s

; OPT-NOT: gvar_unused
; OPTNONE: gvar_unused
@gvar_unused = addrspace(1) global i32 undef, align 4

; ALL: gvar_used
@gvar_used = addrspace(1) global i32 undef, align 4

; OPT: define internal fastcc void @func_used_noinline(
; OPT-NONE: define fastcc void @func_used_noinline(
define fastcc void @func_used_noinline(i32 addrspace(1)* %out, i32 %tid) #1 {
entry:
  store volatile i32 %tid, i32 addrspace(1)* %out
  ret void
}

; OPTNONE: define fastcc void @func_used_alwaysinline(
; OPT-NOT: @func_used_alwaysinline
define fastcc void @func_used_alwaysinline(i32 addrspace(1)* %out, i32 %tid) #2 {
entry:
  store volatile i32 %tid, i32 addrspace(1)* %out
  ret void
}

; OPTNONE: define void @func_unused(
; OPT-NOT: @func_unused
define void @func_unused(i32 addrspace(1)* %out, i32 %tid) #1 {
entry:
  store volatile i32 %tid, i32 addrspace(1)* %out
  ret void
}

; ALL: define amdgpu_kernel void @kernel_unused(
define amdgpu_kernel void @kernel_unused(i32 addrspace(1)* %out) #1 {
entry:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; ALL: define amdgpu_kernel void @main_kernel()
; ALL: tail call i32 @llvm.amdgcn.workitem.id.x
; ALL: tail call fastcc void @func_used_noinline
; ALL: store volatile
; ALL: ret void
define amdgpu_kernel void @main_kernel() {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  tail call fastcc void @func_used_noinline(i32 addrspace(1)* @gvar_used, i32 %tid)
  tail call fastcc void @func_used_alwaysinline(i32 addrspace(1)* @gvar_used, i32 %tid)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { noinline nounwind }
attributes #2 = { alwaysinline nounwind }
