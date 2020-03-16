; XFAIL: windows-gnu,windows-msvc
; NOTE: This is expected to fail on target that do not support memcpy.	
; RUN: llc < %s -mtriple=r600-unknown-linux-gnu 2> %t.err || true	
; RUN: FileCheck --input-file %t.err %s

declare void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define void @test1(i8* %a, i8* %b) nounwind {
; CHECK: LLVM ERROR
  tail call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* %a, i8* %b, i64 8, i1 0 )
  ret void
}
