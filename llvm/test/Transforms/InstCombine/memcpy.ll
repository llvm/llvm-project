; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; Same src/dest.

define void @test1(i8* %a) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    ret void
;
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i32 1, i1 false)
  ret void
}

; PR8267 - same src/dest, but volatile.

define void @test2(i8* %a) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i32 1, i1 true)
; CHECK-NEXT:    ret void
;
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i32 1, i1 true)
  ret void
}

; 17179869184 == 0x400000000 - make sure that doesn't get truncated to 32-bit.

define void @test3(i8* %d, i8* %s) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %d, i8* %s, i64 17179869184, i32 4, i1 false)
; CHECK-NEXT:    ret void
;
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %d, i8* %s, i64 17179869184, i32 4, i1 false)
  ret void
}

