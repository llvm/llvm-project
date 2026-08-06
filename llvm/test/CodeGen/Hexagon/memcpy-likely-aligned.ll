; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test coverage for HexagonSelectionDAGInfo::EmitTargetCodeForMemcpy.
; When a memcpy has known size >= 32, alignment >= 4, and size is a
; multiple of 8, the Hexagon backend should call a specialized memcpy.

; CHECK-LABEL: test_memcpy_aligned:
; CHECK: call __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes
define void @test_memcpy_aligned(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %dst, ptr align 8 %src, i32 64, i1 false)
  ret void
}

; Smaller memcpy (< 32 bytes) should NOT use the specialized path.
; CHECK-LABEL: test_memcpy_small:
; CHECK-NOT: likely_aligned
define void @test_memcpy_small(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 16, i1 false)
  ret void
}

; Non-aligned memcpy should NOT use the specialized path.
; CHECK-LABEL: test_memcpy_unaligned:
; CHECK-NOT: likely_aligned
define void @test_memcpy_unaligned(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %dst, ptr align 1 %src, i32 64, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)

