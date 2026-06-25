; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; A compile-time index past the end of the "VGPR as memory" file is out of range
; (it would otherwise select physical registers outside the reserved file), so
; it is diagnosed rather than miscompiled.

@buf = internal addrspace(13) global [16 x i32] poison

; CHECK: error: {{.*}}unsupported 'VGPR as memory' access: constant index out of range
define amdgpu_kernel void @const_oob() {
  %p = getelementptr i32, ptr addrspace(13) @buf, i32 1000
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(13) @buf
  ret void
}
