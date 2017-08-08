; RUN: llc -march=amdgcn < %s | FileCheck %s

; CHECK-LABEL: {{^}}zext_shl64_to_32:
; CHECK: s_lshl_b32
; CHECK-NOT: s_lshl_b64
define amdgpu_kernel void @zext_shl64_to_32(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 1073741823
  %ext = zext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}sext_shl64_to_32:
; CHECK: s_lshl_b32
; CHECK-NOT: s_lshl_b64
define amdgpu_kernel void @sext_shl64_to_32(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 536870911
  %ext = sext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}zext_shl64_overflow:
; CHECK: s_lshl_b64
; CHECK-NOT: s_lshl_b32
define amdgpu_kernel void @zext_shl64_overflow(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 2147483647
  %ext = zext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}sext_shl64_overflow:
; CHECK: s_lshl_b64
; CHECK-NOT: s_lshl_b32
define amdgpu_kernel void @sext_shl64_overflow(i64 addrspace(1)* nocapture %out, i32 %x) {
  %and = and i32 %x, 2147483647
  %ext = sext i32 %and to i64
  %shl = shl i64 %ext, 2
  store i64 %shl, i64 addrspace(1)* %out, align 4
  ret void
}
