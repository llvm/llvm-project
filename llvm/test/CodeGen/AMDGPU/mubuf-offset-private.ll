; RUN: llc -march=amdgcn -mattr=+max-private-element-size-16 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SICIVI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=+max-private-element-size-16 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SICIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+max-private-element-size-16 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; Test addressing modes when the scratch base is not a frame index.

; GCN-LABEL: {{^}}store_private_offset_i8:
; GCN: buffer_store_byte v{{[0-9]+}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @store_private_offset_i8() #0 {
  store volatile i8 5, i8* inttoptr (i32 8 to i8*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_i16:
; GCN: buffer_store_short v{{[0-9]+}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @store_private_offset_i16() #0 {
  store volatile i16 5, i16* inttoptr (i32 8 to i16*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_i32:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @store_private_offset_i32() #0 {
  store volatile i32 5, i32* inttoptr (i32 8 to i32*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_v2i32:
; GCN: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @store_private_offset_v2i32() #0 {
  store volatile <2 x i32> <i32 5, i32 10>, <2 x i32>* inttoptr (i32 8 to <2 x i32>*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_v4i32:
; GCN: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @store_private_offset_v4i32() #0 {
  store volatile <4 x i32> <i32 5, i32 10, i32 15, i32 0>, <4 x i32>* inttoptr (i32 8 to <4 x i32>*)
  ret void
}

; GCN-LABEL: {{^}}load_private_offset_i8:
; GCN: buffer_load_ubyte v{{[0-9]+}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @load_private_offset_i8() #0 {
  %load = load volatile i8, i8* inttoptr (i32 8 to i8*)
  ret void
}

; GCN-LABEL: {{^}}sextload_private_offset_i8:
; GCN: buffer_load_sbyte v{{[0-9]+}}, off, s[4:7], s8 offset:8
define amdgpu_kernel void @sextload_private_offset_i8(i32 addrspace(1)* %out) #0 {
  %load = load volatile i8, i8* inttoptr (i32 8 to i8*)
  %sextload = sext i8 %load to i32
  store i32 %sextload, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}zextload_private_offset_i8:
; GCN: buffer_load_ubyte v{{[0-9]+}}, off, s[4:7], s8 offset:8
define amdgpu_kernel void @zextload_private_offset_i8(i32 addrspace(1)* %out) #0 {
  %load = load volatile i8, i8* inttoptr (i32 8 to i8*)
  %zextload = zext i8 %load to i32
  store i32 %zextload, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_offset_i16:
; GCN: buffer_load_ushort v{{[0-9]+}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @load_private_offset_i16() #0 {
  %load = load volatile i16, i16* inttoptr (i32 8 to i16*)
  ret void
}

; GCN-LABEL: {{^}}sextload_private_offset_i16:
; GCN: buffer_load_sshort v{{[0-9]+}}, off, s[4:7], s8 offset:8
define amdgpu_kernel void @sextload_private_offset_i16(i32 addrspace(1)* %out) #0 {
  %load = load volatile i16, i16* inttoptr (i32 8 to i16*)
  %sextload = sext i16 %load to i32
  store i32 %sextload, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}zextload_private_offset_i16:
; GCN: buffer_load_ushort v{{[0-9]+}}, off, s[4:7], s8 offset:8
define amdgpu_kernel void @zextload_private_offset_i16(i32 addrspace(1)* %out) #0 {
  %load = load volatile i16, i16* inttoptr (i32 8 to i16*)
  %zextload = zext i16 %load to i32
  store i32 %zextload, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_offset_i32:
; GCN: buffer_load_dword v{{[0-9]+}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @load_private_offset_i32() #0 {
  %load = load volatile i32, i32* inttoptr (i32 8 to i32*)
  ret void
}

; GCN-LABEL: {{^}}load_private_offset_v2i32:
; GCN: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @load_private_offset_v2i32() #0 {
  %load = load volatile <2 x i32>, <2 x i32>* inttoptr (i32 8 to <2 x i32>*)
  ret void
}

; GCN-LABEL: {{^}}load_private_offset_v4i32:
; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[4:7], s2 offset:8
define amdgpu_kernel void @load_private_offset_v4i32() #0 {
  %load = load volatile <4 x i32>, <4 x i32>* inttoptr (i32 8 to <4 x i32>*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_i8_max_offset:
; GCN: buffer_store_byte v{{[0-9]+}}, off, s[4:7], s2 offset:4095
define amdgpu_kernel void @store_private_offset_i8_max_offset() #0 {
  store volatile i8 5, i8* inttoptr (i32 4095 to i8*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_i8_max_offset_plus1:
; GCN: v_mov_b32_e32 [[OFFSET:v[0-9]+]], 0x1000
; GCN: buffer_store_byte v{{[0-9]+}}, [[OFFSET]], s[4:7], s2 offen{{$}}
define amdgpu_kernel void @store_private_offset_i8_max_offset_plus1() #0 {
  store volatile i8 5, i8* inttoptr (i32 4096 to i8*)
  ret void
}

; GCN-LABEL: {{^}}store_private_offset_i8_max_offset_plus2:
; GCN: v_mov_b32_e32 [[OFFSET:v[0-9]+]], 0x1000
; GCN: buffer_store_byte v{{[0-9]+}}, [[OFFSET]], s[4:7], s2 offen offset:1{{$}}
define amdgpu_kernel void @store_private_offset_i8_max_offset_plus2() #0 {
  store volatile i8 5, i8* inttoptr (i32 4097 to i8*)
  ret void
}

; MUBUF used for stack access has bounds checking enabled before gfx9,
; so a possibly negative base index can't be used for the vgpr offset.

; GCN-LABEL: {{^}}store_private_unknown_bits_vaddr:
; SICIVI: v_add_{{i|u}}32_e32 [[ADDR0:v[0-9]+]], vcc, 4
; SICIVI: v_add_{{i|u}}32_e32 [[ADDR1:v[0-9]+]], vcc, 32, [[ADDR0]]
; SICIVI: buffer_store_dword v{{[0-9]+}}, [[ADDR1]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}

; GFX9: v_add_co_u32_e32 [[ADDR:v[0-9]+]], vcc, 4,
; GFX9: buffer_store_dword v{{[0-9]+}}, [[ADDR]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:32
define amdgpu_kernel void @store_private_unknown_bits_vaddr() #0 {
  %alloca = alloca [16 x i32], align 4
  %vaddr = load volatile i32, i32 addrspace(1)* undef
  %vaddr.off = add i32 %vaddr, 8
  %gep = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 %vaddr.off
  store volatile i32 9, i32* %gep
  ret void
}

attributes #0 = { nounwind }
