; RUN: llc -mtriple=aarch64-linux-gnu -stop-after=safe-stack < %s | FileCheck %s

define void @test_sve() safestack {
entry:
  %v = alloca <vscale x 16 x i8>, align 16
  %val = load <vscale x 16 x i8>, ptr %v
  ret void
}

; CHECK-LABEL: define void @test_sve(
; CHECK: [[USP:%.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK: [[SCALE:%.*]] = call i64 @llvm.vscale.i64()
; CHECK: [[SIZE:%.*]] = mul nuw i64 [[SCALE]], 16
; CHECK: [[SP:%.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK: [[NEG_SIZE:%.*]] = sub i64 0, [[SIZE]]
; CHECK: [[NEW_SP:%.*]] = getelementptr i8, ptr [[SP]], i64 [[NEG_SIZE]]
; CHECK: [[ALIGNED_SP:%.*]] = call ptr @llvm.ptrmask.p0.i64(ptr [[NEW_SP]], i64 -16)
; CHECK: store ptr [[ALIGNED_SP]], ptr @__safestack_unsafe_stack_ptr
; CHECK: load <vscale x 16 x i8>, ptr [[ALIGNED_SP]]
; CHECK: store ptr [[USP]], ptr @__safestack_unsafe_stack_ptr
; CHECK: ret void
