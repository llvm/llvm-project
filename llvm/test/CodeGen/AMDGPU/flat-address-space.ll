; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=bonaire < %s | FileCheck -check-prefixes=CHECK,CIVI %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefixes=CHECK,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global < %s | FileCheck -check-prefixes=CHECK,CIVI,CIVI-HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck -check-prefixes=CHECK,GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-flat-for-global < %s | FileCheck -check-prefixes=CHECK,GFX10 %s

; CHECK-LABEL: {{^}}store_flat_i32:
; CHECK-DAG: s_load_dwordx2 s{{\[}}[[LO_SREG:[0-9]+]]:[[HI_SREG:[0-9]+]]],
; CHECK-DAG: s_load_dword s[[SDATA:[0-9]+]],
; CHECK: s_waitcnt lgkmcnt(0)
; CHECK-DAG: v_mov_b32_e32 v[[DATA:[0-9]+]], s[[SDATA]]
; CHECK-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SREG]]
; CHECK-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], s[[HI_SREG]]
; CHECK: flat_store_dword v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}, v[[DATA]]
define amdgpu_kernel void @store_flat_i32(i32 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i32 addrspace(1)* %gptr to i32*
  store volatile i32 %x, i32* %fptr, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_flat_i64:
; CHECK: flat_store_dwordx2
define amdgpu_kernel void @store_flat_i64(i64 addrspace(1)* %gptr, i64 %x) #0 {
  %fptr = addrspacecast i64 addrspace(1)* %gptr to i64*
  store volatile i64 %x, i64* %fptr, align 8
  ret void
}

; CHECK-LABEL: {{^}}store_flat_v4i32:
; CHECK: flat_store_dwordx4
define amdgpu_kernel void @store_flat_v4i32(<4 x i32> addrspace(1)* %gptr, <4 x i32> %x) #0 {
  %fptr = addrspacecast <4 x i32> addrspace(1)* %gptr to <4 x i32>*
  store volatile <4 x i32> %x, <4 x i32>* %fptr, align 16
  ret void
}

; CHECK-LABEL: {{^}}store_flat_trunc_i16:
; CHECK: flat_store_short
define amdgpu_kernel void @store_flat_trunc_i16(i16 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16*
  %y = trunc i32 %x to i16
  store volatile i16 %y, i16* %fptr, align 2
  ret void
}

; CHECK-LABEL: {{^}}store_flat_trunc_i8:
; CHECK: flat_store_byte
define amdgpu_kernel void @store_flat_trunc_i8(i8 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8*
  %y = trunc i32 %x to i8
  store volatile i8 %y, i8* %fptr, align 2
  ret void
}



; CHECK-LABEL: load_flat_i32:
; CHECK: flat_load_dword
define amdgpu_kernel void @load_flat_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i32 addrspace(1)* %gptr to i32*
  %fload = load volatile i32, i32* %fptr, align 4
  store i32 %fload, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: load_flat_i64:
; CHECK: flat_load_dwordx2
define amdgpu_kernel void @load_flat_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i64 addrspace(1)* %gptr to i64*
  %fload = load volatile i64, i64* %fptr, align 8
  store i64 %fload, i64 addrspace(1)* %out, align 8
  ret void
}

; CHECK-LABEL: load_flat_v4i32:
; CHECK: flat_load_dwordx4
define amdgpu_kernel void @load_flat_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast <4 x i32> addrspace(1)* %gptr to <4 x i32>*
  %fload = load volatile <4 x i32>, <4 x i32>* %fptr, align 32
  store <4 x i32> %fload, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; CHECK-LABEL: sextload_flat_i8:
; CHECK: flat_load_sbyte
define amdgpu_kernel void @sextload_flat_i8(i32 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8*
  %fload = load volatile i8, i8* %fptr, align 4
  %ext = sext i8 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: zextload_flat_i8:
; CHECK: flat_load_ubyte
define amdgpu_kernel void @zextload_flat_i8(i32 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8*
  %fload = load volatile i8, i8* %fptr, align 4
  %ext = zext i8 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: sextload_flat_i16:
; CHECK: flat_load_sshort
define amdgpu_kernel void @sextload_flat_i16(i32 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16*
  %fload = load volatile i16, i16* %fptr, align 4
  %ext = sext i16 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: zextload_flat_i16:
; CHECK: flat_load_ushort
define amdgpu_kernel void @zextload_flat_i16(i32 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16*
  %fload = load volatile i16, i16* %fptr, align 4
  %ext = zext i16 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: flat_scratch_unaligned_load:
; CHECK: flat_load_ubyte
; CHECK: flat_load_ubyte
; CHECK: flat_load_ubyte
; CHECK: flat_load_ubyte
define amdgpu_kernel void @flat_scratch_unaligned_load() {
  %scratch = alloca i32, addrspace(5)
  %fptr = addrspacecast i32 addrspace(5)* %scratch to i32*
  %ld = load volatile i32, i32* %fptr, align 1
  ret void
}

; CHECK-LABEL: flat_scratch_unaligned_store:
; CHECK: flat_store_byte
; CHECK: flat_store_byte
; CHECK: flat_store_byte
; CHECK: flat_store_byte
define amdgpu_kernel void @flat_scratch_unaligned_store() {
  %scratch = alloca i32, addrspace(5)
  %fptr = addrspacecast i32 addrspace(5)* %scratch to i32*
  store volatile i32 0, i32* %fptr, align 1
  ret void
}

; CHECK-LABEL: flat_scratch_multidword_load:
; CIVI-HSA: flat_load_dword v
; CIVI-HSA: flat_load_dword v
; GFX9:  flat_load_dwordx2
; GFX10: flat_load_dwordx2
; FIXME: These tests are broken for os = mesa3d, becasue it doesn't initialize flat_scr
define amdgpu_kernel void @flat_scratch_multidword_load() {
  %scratch = alloca <2 x i32>, addrspace(5)
  %fptr = addrspacecast <2 x i32> addrspace(5)* %scratch to <2 x i32>*
  %ld = load volatile <2 x i32>, <2 x i32>* %fptr
  ret void
}

; CHECK-LABEL: flat_scratch_multidword_store:
; CIVI-HSA: flat_store_dword v
; CIVI-HSA: flat_store_dword v
; GFX9:  flat_store_dwordx2
; GFX10: flat_store_dwordx2
; FIXME: These tests are broken for os = mesa3d, becasue it doesn't initialize flat_scr
define amdgpu_kernel void @flat_scratch_multidword_store() {
  %scratch = alloca <2 x i32>, addrspace(5)
  %fptr = addrspacecast <2 x i32> addrspace(5)* %scratch to <2 x i32>*
  store volatile <2 x i32> zeroinitializer, <2 x i32>* %fptr
  ret void
}

; CHECK-LABEL: {{^}}store_flat_i8_max_offset:
; CIVI: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}{{$}}
; GFX9: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:4095{{$}}
define amdgpu_kernel void @store_flat_i8_max_offset(i8* %fptr, i8 %x) #0 {
  %fptr.offset = getelementptr inbounds i8, i8* %fptr, i64 4095
  store volatile i8 %x, i8* %fptr.offset
  ret void
}

; CHECK-LABEL: {{^}}store_flat_i8_max_offset_p1:
; CHECK: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @store_flat_i8_max_offset_p1(i8* %fptr, i8 %x) #0 {
  %fptr.offset = getelementptr inbounds i8, i8* %fptr, i64 4096
  store volatile i8 %x, i8* %fptr.offset
  ret void
}

; CHECK-LABEL: {{^}}store_flat_i8_neg_offset:
; CIVI: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}{{$}}

; GFX9: v_add_co_u32_e64 v{{[0-9]+}}, vcc, -2, s
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, -1,
; GFX9: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @store_flat_i8_neg_offset(i8* %fptr, i8 %x) #0 {
  %fptr.offset = getelementptr inbounds i8, i8* %fptr, i64 -2
  store volatile i8 %x, i8* %fptr.offset
  ret void
}

; CHECK-LABEL: {{^}}load_flat_i8_max_offset:
; CIVI: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GFX9: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} offset:4095 glc{{$}}
; GFX10: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc dlc{{$}}
define amdgpu_kernel void @load_flat_i8_max_offset(i8* %fptr) #0 {
  %fptr.offset = getelementptr inbounds i8, i8* %fptr, i64 4095
  %val = load volatile i8, i8* %fptr.offset
  ret void
}

; CHECK-LABEL: {{^}}load_flat_i8_max_offset_p1:
; CIVI: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GFX9: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GFX10: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc dlc{{$}}
define amdgpu_kernel void @load_flat_i8_max_offset_p1(i8* %fptr) #0 {
  %fptr.offset = getelementptr inbounds i8, i8* %fptr, i64 4096
  %val = load volatile i8, i8* %fptr.offset
  ret void
}

; CHECK-LABEL: {{^}}load_flat_i8_neg_offset:
; CIVI: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc{{$}}

; GFX9: v_add_co_u32_e64 v{{[0-9]+}}, vcc, -2, s
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, -1,
; GFX9: flat_load_ubyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
define amdgpu_kernel void @load_flat_i8_neg_offset(i8* %fptr) #0 {
  %fptr.offset = getelementptr inbounds i8, i8* %fptr, i64 -2
  %val = load volatile i8, i8* %fptr.offset
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
