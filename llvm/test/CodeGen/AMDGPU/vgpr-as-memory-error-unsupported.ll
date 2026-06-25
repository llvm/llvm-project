; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; "VGPR as memory" accesses that the backend cannot lower are diagnosed instead
; of reaching instruction selection as unselectable memory operations.

@buf = internal addrspace(13) global [16 x i64] poison

; A dynamic index into a wider-than-dword element is unsupported.
; CHECK: error: {{.*}}unsupported 'VGPR as memory' access: dynamic index wider than 32 bits
define amdgpu_kernel void @wide_dynamic(i32 %i) {
  %p = getelementptr i64, ptr addrspace(13) @buf, i32 %i
  %v = load i64, ptr addrspace(13) %p
  store i64 %v, ptr addrspace(13) @buf
  ret void
}
