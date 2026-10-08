; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @test_memcpy(ptr %P, ptr %Q, i32 %A, i32 %E) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %E
  ; CHECK-NEXT: call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 1, i32 %E)
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 1, i32 %E)

  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 1, i32 3)

  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 7, i32 4)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr %P, ptr align 4 %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 1 %P, ptr align 4 %Q, i32 4, i32 4)

  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 1 %Q, i32 4, i32 4)

  ret void
}

declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i32) nounwind

define void @test_memmove(ptr %P, ptr %Q, i32 %A, i32 %E) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %E
  ; CHECK-NEXT: call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 1, i32 %E)
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 1, i32 %E)

  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 1, i32 3)

  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 4 %Q, i32 7, i32 4)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr %P, ptr align 4 %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 1 %P, ptr align 4 %Q, i32 4, i32 4)

  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr %Q, i32 1, i32 1)
  ; CHECK: incorrect alignment of the source argument
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 4 %P, ptr align 1 %Q, i32 4, i32 4)

  ret void
}

declare void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i32) nounwind

define void @test_memset(ptr %P, i8 %V, i32 %A, i32 %E) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %E
  ; CHECK: call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 4 %P, i8 %V, i32 1, i32 %E)
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 4 %P, i8 %V, i32 1, i32 %E)

  ; CHECK: element size of the element-wise atomic memory intrinsic must be a power of 2
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 4 %P, i8 %V, i32 1, i32 3)

  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 4 %P, i8 %V, i32 7, i32 4)

  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr %P, i8 %V, i32 1, i32 1)
  ; CHECK: incorrect alignment of the destination argument
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 1 %P, i8 %V, i32 4, i32 4)

  ret void
}
declare void @llvm.memset.element.unordered.atomic.p0.i32(ptr nocapture, i8, i32, i32) nounwind

; The length argument must be a scalar integer.
; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memcpy.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memmove.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)
declare void @llvm.memmove.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memset.element.unordered.atomic.p0.v4i32(ptr, i8, <4 x i32>, i32)
declare void @llvm.memset.element.unordered.atomic.p0.v4i32(ptr, i8, <4 x i32>, i32)

; CHECK: input module is broken!
