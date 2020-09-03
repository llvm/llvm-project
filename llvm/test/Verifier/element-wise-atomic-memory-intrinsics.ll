; RUN: not opt -verify < %s 2>&1 | FileCheck %s

define void @test_memcpy(i8* %P, i8* %Q, i32 %A, i32 %E) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %E
  ; CHECK-NEXT: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1, i32 %E)
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1, i32 %E)

  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1, i32 3)

  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 7, i32 4)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* %P, i8* align 4 %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %P, i8* align 4 %Q, i32 4, i32 4)

  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 1 %Q, i32 4, i32 4)

  ret void
}

declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

define void @test_memmove(i8* %P, i8* %Q, i32 %A, i32 %E) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %E
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1, i32 %E)
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1, i32 %E)

  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 1, i32 3)

  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 4 %Q, i32 7, i32 4)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* %P, i8* align 4 %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 1 %P, i8* align 4 %Q, i32 4, i32 4)

  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* align 4 %P, i8* align 1 %Q, i32 4, i32 4)

  ret void
}

declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

define void @test_memset(i8* %P, i8 %V, i32 %A, i32 %E) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %E
  ; CHECK: call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 4 %P, i8 %V, i32 1, i32 %E)
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 4 %P, i8 %V, i32 1, i32 %E)

  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 4 %P, i8 %V, i32 1, i32 3)

  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 4 %P, i8 %V, i32 7, i32 4)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* %P, i8 %V, i32 1, i32 1)
  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* align 1 %P, i8 %V, i32 4, i32 4)

  ret void
}
declare void @llvm.memset.element.unordered.atomic.p0i8.i32(i8* nocapture, i8, i32, i32) nounwind

; CHECK: input module is broken!
