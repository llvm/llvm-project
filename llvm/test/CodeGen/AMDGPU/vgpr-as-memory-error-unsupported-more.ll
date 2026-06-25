; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; Additional "VGPR as memory" accesses the backend cannot lower, each diagnosed
; rather than reaching instruction selection.

@buf = internal addrspace(13) global [16 x i32] poison

; A dynamic sub-dword access must be naturally aligned so the read-modify-write
; stays within one dword.
; CHECK: error: {{.*}}unsupported 'VGPR as memory' access: underaligned sub-dword dynamic access
define void @underaligned_dyn_subdword(i32 %i, i16 %v) {
  %p = getelementptr i16, ptr addrspace(13) @buf, i32 %i
  store i16 %v, ptr addrspace(13) %p, align 1
  ret void
}

; A dynamic whole-dword access must be dword aligned.
; CHECK: error: {{.*}}unsupported 'VGPR as memory' access: misaligned 32-bit dynamic access
define void @misaligned_dyn_dword(i32 %i, i32 %v) {
  %p = getelementptr i8, ptr addrspace(13) @buf, i32 %i
  %p2 = getelementptr i8, ptr addrspace(13) %p, i32 2
  store i32 %v, ptr addrspace(13) %p2, align 4
  ret void
}

; A constant sub-dword field must not straddle a dword boundary.
; CHECK: error: {{.*}}unsupported 'VGPR as memory' access: sub-dword field crosses a dword boundary
define i16 @const_subdword_crosses_dword() {
  %p = getelementptr i8, ptr addrspace(13) @buf, i32 3
  %v = load i16, ptr addrspace(13) %p, align 1
  ret i16 %v
}
