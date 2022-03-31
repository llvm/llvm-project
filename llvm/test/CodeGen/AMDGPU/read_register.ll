; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

declare i32 @llvm.read_register.i32(metadata) #0
declare i64 @llvm.read_register.i64(metadata) #0

; CHECK-LABEL: {{^}}test_read_m0:
; CHECK: s_mov_b32 m0, -1
; CHECK: v_mov_b32_e32 [[COPY:v[0-9]+]], m0
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[COPY]]
define amdgpu_kernel void @test_read_m0(i32 addrspace(1)* %out) #0 {
  store volatile i32 0, i32 addrspace(3)* undef
  %m0 = call i32 @llvm.read_register.i32(metadata !0)
  store i32 %m0, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_exec:
; CHECK: v_mov_b32_e32 v[[LO:[0-9]+]], exec_lo
; CHECK: v_mov_b32_e32 v[[HI:[0-9]+]], exec_hi
; CHECK: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @test_read_exec(i64 addrspace(1)* %out) #0 {
  %exec = call i64 @llvm.read_register.i64(metadata !1)
  store i64 %exec, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_flat_scratch:
; CHECK: v_mov_b32_e32 v[[LO:[0-9]+]], flat_scratch_lo
; CHECK: v_mov_b32_e32 v[[HI:[0-9]+]], flat_scratch_hi
; CHECK: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @test_read_flat_scratch(i64 addrspace(1)* %out) #0 {
  %flat_scratch = call i64 @llvm.read_register.i64(metadata !2)
  store i64 %flat_scratch, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_flat_scratch_lo:
; CHECK: v_mov_b32_e32 [[COPY:v[0-9]+]], flat_scratch_lo
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[COPY]]
define amdgpu_kernel void @test_read_flat_scratch_lo(i32 addrspace(1)* %out) #0 {
  %flat_scratch_lo = call i32 @llvm.read_register.i32(metadata !3)
  store i32 %flat_scratch_lo, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_flat_scratch_hi:
; CHECK: v_mov_b32_e32 [[COPY:v[0-9]+]], flat_scratch_hi
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[COPY]]
define amdgpu_kernel void @test_read_flat_scratch_hi(i32 addrspace(1)* %out) #0 {
  %flat_scratch_hi = call i32 @llvm.read_register.i32(metadata !4)
  store i32 %flat_scratch_hi, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_exec_lo:
; CHECK: v_mov_b32_e32 [[COPY:v[0-9]+]], exec_lo
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[COPY]]
define amdgpu_kernel void @test_read_exec_lo(i32 addrspace(1)* %out) #0 {
  %exec_lo = call i32 @llvm.read_register.i32(metadata !5)
  store i32 %exec_lo, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}test_read_exec_hi:
; CHECK: v_mov_b32_e32 [[COPY:v[0-9]+]], exec_hi
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[COPY]]
define amdgpu_kernel void @test_read_exec_hi(i32 addrspace(1)* %out) #0 {
  %exec_hi = call i32 @llvm.read_register.i32(metadata !6)
  store i32 %exec_hi, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"m0"}
!1 = !{!"exec"}
!2 = !{!"flat_scratch"}
!3 = !{!"flat_scratch_lo"}
!4 = !{!"flat_scratch_hi"}
!5 = !{!"exec_lo"}
!6 = !{!"exec_hi"}
