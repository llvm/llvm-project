; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1210 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1210 -verify-machineinstrs < %s | FileCheck -check-prefix=MESA %s

; GFX1210 supports upto 384 KB configurable LDS memory.
; This test checks the min and max size of LDS that can be allocated.

@lds.i8 = addrspace(3) global i8 undef
@lds.array.i8 = addrspace(3) global [393215 x i8] undef
@lds.i16 = addrspace(3) global i16 undef
@lds.array.i16 = addrspace(3) global [196607 x i16] undef
@lds.i32 = addrspace(3) global i32 undef
@lds.array.i32 = addrspace(3) global [98303 x i32] undef

; GCN-LABEL: test_lds_i8:
; GCN: .amdhsa_group_segment_fixed_size 1
; GCN: ; LDSByteSize: 1 bytes/workgroup
; MESA: granulated_lds_size = 1
define amdgpu_kernel void @test_lds_i8(i8 %val) {
  store i8 %val, ptr addrspace(3) @lds.i8
  ret void
}

; GCN-LABEL: test_lds_i16:
; GCN: .amdhsa_group_segment_fixed_size 2
; GCN: ; LDSByteSize: 2 bytes/workgroup
; MESA: granulated_lds_size = 1
define amdgpu_kernel void @test_lds_i16(i16 %val) {
  store i16 %val, ptr addrspace(3) @lds.i16
  ret void
}

; GCN-LABEL: test_lds_i32:
; GCN: .amdhsa_group_segment_fixed_size 4
; GCN: ; LDSByteSize: 4 bytes/workgroup
; MESA: granulated_lds_size = 1
define amdgpu_kernel void @test_lds_i32(i32 %val) {
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}

; GCN-LABEL: test_lds_array_i8:
; GCN: .amdhsa_group_segment_fixed_size 393216
; GCN: ; LDSByteSize: 393216 bytes/workgroup
; MESA: granulated_lds_size = 384
define amdgpu_kernel void @test_lds_array_i8() {
  %gep = getelementptr inbounds [393215 x i8], ptr addrspace(3) @lds.array.i8, i32 0, i32 5
  %val = load i8, ptr addrspace(3) %gep
  store i8 %val, ptr addrspace(3) @lds.i8
  ret void
}

; GCN-LABEL: test_lds_array_i16:
; GCN: .amdhsa_group_segment_fixed_size 393216
; GCN: ; LDSByteSize: 393216 bytes/workgroup
; MESA: granulated_lds_size = 384
define amdgpu_kernel void @test_lds_array_i16() {
  %gep = getelementptr inbounds [196607 x i16], ptr addrspace(3) @lds.array.i16, i32 0, i32 10
  %val = load i16, ptr addrspace(3) %gep
  store i16 %val, ptr addrspace(3) @lds.i16
  ret void
}

; GCN-LABEL: test_lds_array_i32:
; GCN: .amdhsa_group_segment_fixed_size 393216
; GCN: ; LDSByteSize: 393216 bytes/workgroup
; MESA: granulated_lds_size = 384
define amdgpu_kernel void @test_lds_array_i32() {
  %gep = getelementptr inbounds [98303 x i32], ptr addrspace(3) @lds.array.i32, i32 0, i32 20
  %val = load i32, ptr addrspace(3) %gep
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}
