; NOTE: This is expected to fail on target that do not support memcpy.	
; RUN: llc < %s -mtriple=r600-unknown-linux-gnu 2> %t.err || true	
; RUN: FileCheck --input-file %t.err %s

declare void @llvm.memcpy.inline.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

define void @test1(ptr %a, ptr %b) nounwind {
; CHECK: LLVM ERROR
  tail call void @llvm.memcpy.inline.p0.p0.i64(ptr %a, ptr %b, i64 8, i1 0 )
  ret void
}
