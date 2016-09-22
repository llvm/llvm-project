; RUN: llc -mtriple=aarch64-apple-ios -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s


; CHECK-LABEL: name: test_stack_slots
; CHECK: fixedStack:
; CHECK-DAG:  - { id: [[STACK0:[0-9]+]], offset: 0, size: 1
; CHECK-DAG:  - { id: [[STACK8:[0-9]+]], offset: 1, size: 1
; CHECK: [[LHS_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; CHECK: [[LHS:%[0-9]+]](s8) = G_LOAD [[LHS_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK0]], align 0)
; CHECK: [[RHS_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; CHECK: [[RHS:%[0-9]+]](s8) = G_LOAD [[RHS_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK8]], align 0)
; CHECK: [[SUM:%[0-9]+]](s8) = G_ADD [[LHS]], [[RHS]]
; CHECK: [[SUM32:%[0-9]+]](s32) = G_SEXT [[SUM]](s8)
; CHECK: %w0 = COPY [[SUM32]](s32)
define signext i8 @test_stack_slots([8 x i64], i8 signext %lhs, i8 signext %rhs) {
  %sum = add i8 %lhs, %rhs
  ret i8 %sum
}

; CHECK-LABEL: name: test_call_stack
; CHECK: [[C42:%[0-9]+]](s8) = G_CONSTANT 42
; CHECK: [[C12:%[0-9]+]](s8) = G_CONSTANT 12
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[C42_OFFS:%[0-9]+]](s64) = G_CONSTANT 0
; CHECK: [[C42_LOC:%[0-9]+]](p0) = G_GEP [[SP]], [[C42_OFFS]](s64)
; CHECK: G_STORE [[C42]](s8), [[C42_LOC]](p0) :: (store 1 into stack, align 0)
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[C12_OFFS:%[0-9]+]](s64) = G_CONSTANT 1
; CHECK: [[C12_LOC:%[0-9]+]](p0) = G_GEP [[SP]], [[C12_OFFS]](s64)
; CHECK: G_STORE [[C12]](s8), [[C12_LOC]](p0) :: (store 1 into stack + 1, align 0)
; CHECK: BL @test_stack_slots
define void @test_call_stack() {
  call signext i8 @test_stack_slots([8 x i64] undef, i8 signext 42, i8 signext 12)
  ret void
}
