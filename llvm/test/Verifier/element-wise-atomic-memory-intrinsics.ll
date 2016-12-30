; RUN: not opt -verify < %s 2>&1 | FileCheck %s

define void @test_memcpy(i8* %P, i8* %Q) {
  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 2 %P, i8* align 2 %Q, i64 4, i32 3)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 2 %P, i8* align 4 %Q, i64 4, i32 4)

  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %P, i8* align 2 %Q, i64 4, i32 4)

  ret void
}
declare void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* nocapture, i8* nocapture, i64, i32) nounwind

; CHECK: input module is broken!
