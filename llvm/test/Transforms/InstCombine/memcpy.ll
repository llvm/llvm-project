; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define void @test1(i8* %a) {
        tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i1 false)
        ret void
; CHECK-LABEL: define void @test1(
; CHECK-NEXT: ret void
}


; PR8267
define void @test2(i8* %a) {
        tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i1 true)
        ret void
; CHECK-LABEL: define void @test2(
; CHECK-NEXT: call void @llvm.memcpy
}

define void @test3(i8* %d, i8* %s) {
        tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %d, i8* %s, i64 17179869184, i1 false)
        ret void
; CHECK-LABEL: define void @test3(
; CHECK-NEXT: call void @llvm.memcpy
}
