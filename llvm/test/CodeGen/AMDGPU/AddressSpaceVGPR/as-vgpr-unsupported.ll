; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=error:

; Unimplemented accesses get a clean diagnostic on both selectors, not a
; selection or legalization failure, or wrong code.

; A type that is not a whole number of bytes is not implemented.
; CHECK: error: {{.*}}in function load_i1 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only loads and stores of whole bytes are implemented
define i1 @load_i1(ptr addrspace(13) inreg %p) {
  %x = load i1, ptr addrspace(13) %p
  ret i1 %x
}

; CHECK: error: {{.*}}in function store_i1 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only loads and stores of whole bytes are implemented
define void @store_i1(ptr addrspace(13) inreg %p, i1 %v) {
  store i1 %v, ptr addrspace(13) %p
  ret void
}

; CHECK: error: {{.*}}in function load_i4 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only loads and stores of whole bytes are implemented
define i4 @load_i4(ptr addrspace(13) inreg %p) {
  %x = load i4, ptr addrspace(13) %p
  ret i4 %x
}

; CHECK: error: {{.*}}in function store_v4i1 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only loads and stores of whole bytes are implemented
define void @store_v4i1(ptr addrspace(13) inreg %p, <4 x i1> %v) {
  store <4 x i1> %v, ptr addrspace(13) %p
  ret void
}

; CHECK: error: {{.*}}in function load_i12 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only loads and stores of whole bytes are implemented
define i12 @load_i12(ptr addrspace(13) inreg %p) {
  %x = load i12, ptr addrspace(13) %p
  ret i12 %x
}

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
