; RUN: opt -S < %s | FileCheck %s

; Test to ensure that calls to the memcpy/memmove/memset intrinsics are auto-upgraded
; to remove the alignment parameter in favour of align attributes on the pointer args.

; Make sure a non-zero alignment is propagated
define void @test(i8* %p1, i8* %p2, i8* %p3) {
; CHECK-LABEL: @test
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %p1, i8 55, i64 100, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %p1, i8* align 4 %p2, i64 50, i1 false)
; CHECK: call void @llvm.memmove.p0i8.p0i8.i64(i8* align 4 %p2, i8* align 4 %p3, i64 1000, i1 false)
  call void @llvm.memset.p0i8.i64(i8* %p1, i8 55, i64 100, i32 4, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p1, i8* %p2, i64 50, i32 4, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %p2, i8* %p3, i64 1000, i32 4, i1 false)
  ret void
}

; Make sure that a zero alignment is handled properly
define void @test2(i8* %p1, i8* %p2, i8* %p3) {
; CHECK-LABEL: @test
; CHECK: call void @llvm.memset.p0i8.i64(i8* %p1, i8 55, i64 100, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p1, i8* %p2, i64 50, i1 false)
; CHECK: call void @llvm.memmove.p0i8.p0i8.i64(i8* %p2, i8* %p3, i64 1000, i1 false)
  call void @llvm.memset.p0i8.i64(i8* %p1, i8 55, i64 100, i32 0, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p1, i8* %p2, i64 50, i32 0, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %p2, i8* %p3, i64 1000, i32 0, i1 false)
  ret void
}

; CHECK: declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)
; CHECK: declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg)
; CHECK: declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly , i8* nocapture readonly, i64, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1)

