; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null < %s 2>&1 | FileCheck %s --implicit-check-not=error:

; Unimplemented accesses get a clean diagnostic on both selectors, not a
; selection or legalization failure, or wrong code.

; Sub-dword accesses are not implemented yet.
; CHECK: error: {{.*}}in function load_i8 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i8 @load_i8(ptr addrspace(13) inreg %p) {
  %x = load i8, ptr addrspace(13) %p
  ret i8 %x
}

; CHECK: error: {{.*}}in function load_i16 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i16 @load_i16(ptr addrspace(13) inreg %p) {
  %x = load i16, ptr addrspace(13) %p
  ret i16 %x
}

; CHECK: error: {{.*}}in function store_i8 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_i8(ptr addrspace(13) inreg %p, i8 %v) {
  store i8 %v, ptr addrspace(13) %p
  ret void
}

; CHECK: error: {{.*}}in function store_i16 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_i16(ptr addrspace(13) inreg %p, i16 %v) {
  store i16 %v, ptr addrspace(13) %p
  ret void
}

; So are accesses past a dword that are not whole dwords.
; CHECK: error: {{.*}}in function load_i48 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i48 @load_i48(ptr addrspace(13) inreg %p) {
  %x = load i48, ptr addrspace(13) %p, align 4
  ret i48 %x
}

; CHECK: error: {{.*}}in function store_v3i16 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_v3i16(ptr addrspace(13) inreg %p, <3 x i16> %v) {
  store <3 x i16> %v, ptr addrspace(13) %p, align 4
  ret void
}

; So are the extending loads both combiners make of these.
; CHECK: error: {{.*}}in function zextload_i16 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i32 @zextload_i16(ptr addrspace(13) inreg %p) {
  %x = load i16, ptr addrspace(13) %p
  %z = zext i16 %x to i32
  ret i32 %z
}

; CHECK: error: {{.*}}in function sextload_i16 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i32 @sextload_i16(ptr addrspace(13) inreg %p) {
  %x = load i16, ptr addrspace(13) %p
  %s = sext i16 %x to i32
  ret i32 %s
}

; The index is the pointer >> 2, so an under-aligned access would silently reach
; the containing dword.
; CHECK: error: {{.*}}in function load_i32_align1 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define i32 @load_i32_align1(ptr addrspace(13) inreg %p) {
  %x = load i32, ptr addrspace(13) %p, align 1
  ret i32 %x
}

; CHECK: error: {{.*}}in function store_i32_align1 {{.*}}: unsupported access of VGPR 'as memory' address space (13); only dword-aligned whole-dword loads and stores are implemented
define void @store_i32_align1(ptr addrspace(13) inreg %p, i32 %v) {
  store i32 %v, ptr addrspace(13) %p, align 1
  ret void
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
