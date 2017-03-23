; RUN: llc -O0 -mtriple=amdgcn-mesa-mesa3d -mcpu=bonaire < %s | FileCheck  %s
; RUN: llc -O0 -mtriple=amdgcn-mesa-mesa3d -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck  %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global < %s | FileCheck -check-prefixes=CHECK,HSA %s

; Disable optimizations in case there are optimizations added that
; specialize away generic pointer accesses.


; These testcases might become useless when there are optimizations to
; remove generic pointers.

; CHECK-LABEL: {{^}}store_flat_i32:
; CHECK-DAG: s_load_dwordx2 s{{\[}}[[LO_SREG:[0-9]+]]:[[HI_SREG:[0-9]+]]],
; CHECK-DAG: s_load_dword s[[SDATA:[0-9]+]],
; CHECK: s_waitcnt lgkmcnt(0)
; CHECK-DAG: v_mov_b32_e32 v[[DATA:[0-9]+]], s[[SDATA]]
; CHECK-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SREG]]
; CHECK-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], s[[HI_SREG]]
; CHECK: flat_store_dword v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}, v[[DATA]]
define amdgpu_kernel void @store_flat_i32(i32 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i32 addrspace(1)* %gptr to i32 addrspace(4)*
  store volatile i32 %x, i32 addrspace(4)* %fptr, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_flat_i64:
; CHECK: flat_store_dwordx2
define amdgpu_kernel void @store_flat_i64(i64 addrspace(1)* %gptr, i64 %x) #0 {
  %fptr = addrspacecast i64 addrspace(1)* %gptr to i64 addrspace(4)*
  store volatile i64 %x, i64 addrspace(4)* %fptr, align 8
  ret void
}

; CHECK-LABEL: {{^}}store_flat_v4i32:
; CHECK: flat_store_dwordx4
define amdgpu_kernel void @store_flat_v4i32(<4 x i32> addrspace(1)* %gptr, <4 x i32> %x) #0 {
  %fptr = addrspacecast <4 x i32> addrspace(1)* %gptr to <4 x i32> addrspace(4)*
  store volatile <4 x i32> %x, <4 x i32> addrspace(4)* %fptr, align 16
  ret void
}

; CHECK-LABEL: {{^}}store_flat_trunc_i16:
; CHECK: flat_store_short
define amdgpu_kernel void @store_flat_trunc_i16(i16 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16 addrspace(4)*
  %y = trunc i32 %x to i16
  store volatile i16 %y, i16 addrspace(4)* %fptr, align 2
  ret void
}

; CHECK-LABEL: {{^}}store_flat_trunc_i8:
; CHECK: flat_store_byte
define amdgpu_kernel void @store_flat_trunc_i8(i8 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8 addrspace(4)*
  %y = trunc i32 %x to i8
  store volatile i8 %y, i8 addrspace(4)* %fptr, align 2
  ret void
}



; CHECK-LABEL: load_flat_i32:
; CHECK: flat_load_dword
define amdgpu_kernel void @load_flat_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i32 addrspace(1)* %gptr to i32 addrspace(4)*
  %fload = load volatile i32, i32 addrspace(4)* %fptr, align 4
  store i32 %fload, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: load_flat_i64:
; CHECK: flat_load_dwordx2
define amdgpu_kernel void @load_flat_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i64 addrspace(1)* %gptr to i64 addrspace(4)*
  %fload = load volatile i64, i64 addrspace(4)* %fptr, align 8
  store i64 %fload, i64 addrspace(1)* %out, align 8
  ret void
}

; CHECK-LABEL: load_flat_v4i32:
; CHECK: flat_load_dwordx4
define amdgpu_kernel void @load_flat_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast <4 x i32> addrspace(1)* %gptr to <4 x i32> addrspace(4)*
  %fload = load volatile <4 x i32>, <4 x i32> addrspace(4)* %fptr, align 32
  store <4 x i32> %fload, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; CHECK-LABEL: sextload_flat_i8:
; CHECK: flat_load_sbyte
define amdgpu_kernel void @sextload_flat_i8(i32 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8 addrspace(4)*
  %fload = load volatile i8, i8 addrspace(4)* %fptr, align 4
  %ext = sext i8 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: zextload_flat_i8:
; CHECK: flat_load_ubyte
define amdgpu_kernel void @zextload_flat_i8(i32 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8 addrspace(4)*
  %fload = load volatile i8, i8 addrspace(4)* %fptr, align 4
  %ext = zext i8 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: sextload_flat_i16:
; CHECK: flat_load_sshort
define amdgpu_kernel void @sextload_flat_i16(i32 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16 addrspace(4)*
  %fload = load volatile i16, i16 addrspace(4)* %fptr, align 4
  %ext = sext i16 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: zextload_flat_i16:
; CHECK: flat_load_ushort
define amdgpu_kernel void @zextload_flat_i16(i32 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16 addrspace(4)*
  %fload = load volatile i16, i16 addrspace(4)* %fptr, align 4
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
  %scratch = alloca i32
  %fptr = addrspacecast i32* %scratch to i32 addrspace(4)*
  %ld = load volatile i32, i32 addrspace(4)* %fptr, align 1
  ret void
}

; CHECK-LABEL: flat_scratch_unaligned_store:
; CHECK: flat_store_byte
; CHECK: flat_store_byte
; CHECK: flat_store_byte
; CHECK: flat_store_byte
define amdgpu_kernel void @flat_scratch_unaligned_store() {
  %scratch = alloca i32
  %fptr = addrspacecast i32* %scratch to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %fptr, align 1
  ret void
}

; CHECK-LABEL: flat_scratch_multidword_load:
; HSA: flat_load_dword
; HSA: flat_load_dword
; FIXME: These tests are broken for os = mesa3d, becasue it doesn't initialize flat_scr
define amdgpu_kernel void @flat_scratch_multidword_load() {
  %scratch = alloca <2 x i32>
  %fptr = addrspacecast <2 x i32>* %scratch to <2 x i32> addrspace(4)*
  %ld = load volatile <2 x i32>, <2 x i32> addrspace(4)* %fptr
  ret void
}

; CHECK-LABEL: flat_scratch_multidword_store:
; HSA: flat_store_dword
; HSA: flat_store_dword
; FIXME: These tests are broken for os = mesa3d, becasue it doesn't initialize flat_scr
define amdgpu_kernel void @flat_scratch_multidword_store() {
  %scratch = alloca <2 x i32>
  %fptr = addrspacecast <2 x i32>* %scratch to <2 x i32> addrspace(4)*
  store volatile <2 x i32> zeroinitializer, <2 x i32> addrspace(4)* %fptr
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #3 = { nounwind readnone }
