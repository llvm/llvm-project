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
