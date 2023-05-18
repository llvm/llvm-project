; RUN: llc -march=mips -mcpu=mips32r2 < %s -o /dev/null

; Test that SelectionDAG does not crash during DAGCombine when two pointers
; to the stack match with differing bases and offsets when expanding memcpy.
; This could result in one of the pointers being considered dereferenceable
; and other not.

define void @foo(ptr) {
start:
  %a = alloca [22 x i8]
  %b = alloca [22 x i8]
  %d = getelementptr inbounds [22 x i8], ptr %b, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %d, i32 20, i1 false)
  %e = getelementptr inbounds [22 x i8], ptr %b, i32 0, i32 6
  call void @llvm.memcpy.p0.p0.i32(ptr %0, ptr %e, i32 12, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)
