; RUN: llc -march=mips -mcpu=mips32r2 < %s -o /dev/null

; Test that SelectionDAG does not crash during DAGCombine when two pointers
; to the stack match with differing bases and offsets when expanding memcpy.
; This could result in one of the pointers being considered dereferenceable
; and other not.

define void @foo(i8*) {
start:
  %a = alloca [22 x i8]
  %b = alloca [22 x i8]
  %c = bitcast [22 x i8]* %a to i8*
  %d = getelementptr inbounds [22 x i8], [22 x i8]* %b, i32 0, i32 2
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %c, i8* %d, i32 20, i32 1, i1 false)
  %e = getelementptr inbounds [22 x i8], [22 x i8]* %b, i32 0, i32 6
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %e, i32 12, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)
