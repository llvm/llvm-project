; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -asm-verbose=0 < %s | FileCheck %s

define amdgpu_kernel void @range_with_padding(i64 %unused, i32 inreg %hot0, i64 inreg %hot1, i32 %unused2) #0 {
; CHECK-LABEL: range_with_padding:
; CHECK: s_load_dwordx4 {{.*}}, 0x8
; CHECK: .amdhsa_kernel range_with_padding
; CHECK: .amdhsa_user_sgpr_kernarg_preload_length 4
; CHECK: .amdhsa_user_sgpr_kernarg_preload_offset 2
  ret void
}

define amdgpu_kernel void @subdword_range_start(i8 %unused, i8 inreg %hot, ptr addrspace(1) %out) #1 {
; CHECK-LABEL: subdword_range_start:
; CHECK: s_load_dword {{.*}}, 0x0
; CHECK: .amdhsa_kernel subdword_range_start
; CHECK: .amdhsa_user_sgpr_kernarg_preload_length 1
; CHECK: .amdhsa_user_sgpr_kernarg_preload_offset 0
  store i8 %hot, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @range_after_byref(ptr addrspace(4) byref([8 x i32]) align 32 %unused, i32 inreg %hot) #1 {
; CHECK-LABEL: range_after_byref:
; CHECK: s_load_dword {{.*}}, 0x20
; CHECK: .amdhsa_kernel range_after_byref
; CHECK: .amdhsa_user_sgpr_kernarg_preload_length 1
; CHECK: .amdhsa_user_sgpr_kernarg_preload_offset 8
  ret void
}

attributes #0 = { "amdgpu-kernarg-preload-first-arg"="1" "amdgpu-kernarg-preload-last-arg"="2" }
attributes #1 = { "amdgpu-kernarg-preload-first-arg"="1" "amdgpu-kernarg-preload-last-arg"="1" }
