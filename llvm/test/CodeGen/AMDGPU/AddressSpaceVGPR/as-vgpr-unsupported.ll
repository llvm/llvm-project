; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=error:

; What is not implemented gets a clean diagnostic on both selectors, not a
; selection or legalization failure, or wrong code.

; An alloca in VGPR "as memory" has no registers to live in, so one that
; reaches instruction selection is diagnosed rather than given a stack slot
; whose offset would be taken for a register address.
; CHECK: error: {{.*}}in function alloca_vgpr {{.*}}: unsupported alloca in VGPR 'as memory' address space (13)
define void @alloca_vgpr(ptr addrspace(1) %out, i32 %v) {
  %a = alloca i32, align 4, addrspace(13)
  store i32 %v, ptr addrspace(13) %a, align 4
  %x = load i32, ptr addrspace(13) %a, align 4
  store i32 %x, ptr addrspace(1) %out
  ret void
}

declare void @use(ptr addrspace(13))

; CHECK: error: {{.*}}in function alloca_vgpr_escapes {{.*}}: unsupported alloca in VGPR 'as memory' address space (13)
define void @alloca_vgpr_escapes() {
  %a = alloca [4 x i32], align 4, addrspace(13)
  call void @use(ptr addrspace(13) %a)
  ret void
}

declare void @use_private(ptr addrspace(5))

define void @alloca_private() {
  %a = alloca [4 x i32], align 4, addrspace(5)
  call void @use_private(ptr addrspace(5) %a)
  ret void
}
