; RUN: llc -mtriple=aarch64-linux-gnu -stop-after=safe-stack < %s | FileCheck %s

define void @test_sve() safestack {
entry:
  %v = alloca <vscale x 16 x i8>, align 16
  %val = load <vscale x 16 x i8>, ptr %v
  ret void
}

; CHECK-LABEL: define void @test_sve(
; CHECK: [[USP:%.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK: [[USST:%.*]] = getelementptr i8, ptr [[USP]], i32 -16
; CHECK: store ptr [[USST]], ptr @__safestack_unsafe_stack_ptr
; CHECK: [[PTR:%.*]] = getelementptr i8, ptr [[USP]], i32 -16
; CHECK: load <vscale x 16 x i8>, ptr [[PTR]]
; CHECK: store ptr [[USP]], ptr @__safestack_unsafe_stack_ptr
; CHECK: ret void

declare void @escape(ptr)

; A byval argument of a scalable type has no size known at compile time, so it
; cannot be placed on the unsafe stack frame and is left alone.

define void @test_sve_byval(ptr byval(<vscale x 4 x i32>) %p) safestack {
  call void @escape(ptr %p)
  ret void
}

; CHECK-LABEL: define void @test_sve_byval(
; CHECK-NEXT: call void @escape(ptr %p)
; CHECK-NEXT: ret void

; A byval argument of a fixed size is still copied onto the unsafe stack.

define void @test_byval(ptr byval([16 x i8]) %p) safestack {
  call void @escape(ptr %p)
  ret void
}

; CHECK-LABEL: define void @test_byval(
; CHECK: load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[SLOT:%.*]], ptr %p, i64 16, i1 false)
; CHECK: call void @escape(ptr [[SLOT]])
