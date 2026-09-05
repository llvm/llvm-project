; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s

declare i32 @llvm.read_register.i32(metadata) #0
declare i64 @llvm.read_register.i64(metadata) #0

; CHECK-LABEL: {{^}}test_read_src_flat_scratch_base:
; CHECK: v_mov_b64_e32 v{{\[[0-9]+:[0-9]+\]}}, src_flat_scratch_base_lo
define amdgpu_kernel void @test_read_src_flat_scratch_base(ptr addrspace(1) %out) #0 {
  %v = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %v, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_src_flat_scratch_base_lo:
; CHECK: v_{{(dual_)?}}mov_b32{{(_e32)?}} v{{[0-9]+}}, src_flat_scratch_base_lo
define amdgpu_kernel void @test_read_src_flat_scratch_base_lo(ptr addrspace(1) %out) #0 {
  %v = call i32 @llvm.read_register.i32(metadata !1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_src_flat_scratch_base_hi:
; CHECK: v_{{(dual_)?}}mov_b32{{(_e32)?}} v{{[0-9]+}}, src_flat_scratch_base_hi
define amdgpu_kernel void @test_read_src_flat_scratch_base_hi(ptr addrspace(1) %out) #0 {
  %v = call i32 @llvm.read_register.i32(metadata !2)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Reading into an SGPR goes through a copy, now that the registers are no
; longer members of the SGPR register classes.

; CHECK-LABEL: {{^}}test_read_src_flat_scratch_base_sgpr:
; CHECK: s_mov_b64 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, src_flat_scratch_base_lo
; CHECK: ; use s{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_read_src_flat_scratch_base_sgpr() #0 {
  %v = call i64 @llvm.read_register.i64(metadata !0)
  call void asm sideeffect "; use $0", "s"(i64 %v)
  ret void
}

; CHECK-LABEL: {{^}}test_read_src_flat_scratch_base_lo_sgpr:
; CHECK: s_mov_b32 [[REG:s[0-9]+]], src_flat_scratch_base_lo
; CHECK: ; use [[REG]]
define amdgpu_kernel void @test_read_src_flat_scratch_base_lo_sgpr() #0 {
  %v = call i32 @llvm.read_register.i32(metadata !1)
  call void asm sideeffect "; use $0", "s"(i32 %v)
  ret void
}

; CHECK-LABEL: {{^}}test_read_src_flat_scratch_base_hi_sgpr:
; CHECK: s_mov_b32 [[REG:s[0-9]+]], src_flat_scratch_base_hi
; CHECK: ; use [[REG]]
define amdgpu_kernel void @test_read_src_flat_scratch_base_hi_sgpr() #0 {
  %v = call i32 @llvm.read_register.i32(metadata !2)
  call void asm sideeffect "; use $0", "s"(i32 %v)
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"src_flat_scratch_base"}
!1 = !{!"src_flat_scratch_base_lo"}
!2 = !{!"src_flat_scratch_base_hi"}
