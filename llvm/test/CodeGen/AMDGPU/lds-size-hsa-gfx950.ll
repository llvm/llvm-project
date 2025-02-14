; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck -check-prefix=MESA %s

; gfx950 supports upto 160 KB configurable LDS memory.
; This test checks the max and above the old i.e. 128 KiB size of LDS that can be allocated.

@lds.i32 = addrspace(3) global i32 poison
@lds.array.size.131076 = addrspace(3) global [32768 x i32] poison
@lds.array.size.163840 = addrspace(3) global [40959 x i32] poison

; GCN-LABEL: test_lds_array_size_131076:
; GCN: .amdhsa_group_segment_fixed_size 131076
; GCN: ; LDSByteSize: 131076 bytes/workgroup
; MESA: granulated_lds_size = 65
define amdgpu_kernel void @test_lds_array_size_131076() {
  %gep = getelementptr inbounds [32768 x i32], ptr addrspace(3) @lds.array.size.131076, i32 0, i32 20
  %val = load i32, ptr addrspace(3) %gep
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}

; GCN-LABEL: test_lds_array_size_163840:
; GCN: .amdhsa_group_segment_fixed_size 163840
; GCN: ; LDSByteSize: 163840 bytes/workgroup
; MESA: granulated_lds_size = 80
define amdgpu_kernel void @test_lds_array_size_163840() {
  %gep = getelementptr inbounds [40959 x i32], ptr addrspace(3) @lds.array.size.163840 , i32 0, i32 20
  %val = load i32, ptr addrspace(3) %gep
  store i32 %val, ptr addrspace(3) @lds.i32
  ret void
}
