; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; A dynamic index addresses the whole "VGPR as memory" file as one indexed
; tuple. A file whose (even-dword-rounded) size has no VGPR tuple class - e.g.
; 14 dwords - is diagnosed rather than aborting the compiler.

@buf = internal addrspace(13) global [14 x i32] poison

; CHECK: error: {{.*}}unsupported 'VGPR as memory' access: VGPR-memory file too large for a dynamic index
define amdgpu_kernel void @dynamic_toolarge(i32 %i) {
  %p = getelementptr i32, ptr addrspace(13) @buf, i32 %i
  %v = load i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(13) @buf
  ret void
}
