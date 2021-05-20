; RUN: llc -mtriple=aarch64-apple-ios -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s


; CHECK-LABEL: name: test_stack_slots
; CHECK: fixedStack:
; CHECK-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 1,
; CHECK-DAG:  - { id: [[STACK8:[0-9]+]], type: default, offset: 1, size: 1,
; CHECK:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
; CHECK:   [[LOAD:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX]](p0) :: (invariant load (s8) from %fixed-stack.1, align 16)
; CHECK:   [[ASSERT_SEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[LOAD]], 8
; CHECK:   [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[ASSERT_SEXT]](s32)
; CHECK:   [[FRAME_INDEX1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
; CHECK:   [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX1]](p0) :: (invariant load (s8) from %fixed-stack.0)
; CHECK:   [[ASSERT_SEXT1:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[LOAD1]], 8
; CHECK:   [[TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[ASSERT_SEXT1]](s32)
; CHECK:   [[ADD:%[0-9]+]]:_(s8) = G_ADD [[TRUNC]], [[TRUNC1]]
; CHECK:   [[SEXT:%[0-9]+]]:_(s32) = G_SEXT [[ADD]](s8)
; CHECK:   $w0 = COPY [[SEXT]](s32)
; CHECK:   RET_ReallyLR implicit $w0
define signext i8 @test_stack_slots([8 x i64], i8 signext %lhs, i8 signext %rhs) {
  %sum = add i8 %lhs, %rhs
  ret i8 %sum
}

; CHECK-LABEL: name: test_call_stack
; CHECK: [[C42:%[0-9]+]]:_(s8) = G_CONSTANT i8 42
; CHECK: [[C12:%[0-9]+]]:_(s8) = G_CONSTANT i8 12
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[C42_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[C42_LOC:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[C42_OFFS]](s64)
; CHECK: G_STORE [[C42]](s8), [[C42_LOC]](p0) :: (store (s8) into stack)
; CHECK: [[C12_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: [[C12_LOC:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[C12_OFFS]](s64)
; CHECK: G_STORE [[C12]](s8), [[C12_LOC]](p0) :: (store (s8) into stack + 1)
; CHECK: BL @test_stack_slots
define void @test_call_stack() {
  call signext i8 @test_stack_slots([8 x i64] undef, i8 signext 42, i8 signext 12)
  ret void
}

; CHECK-LABEL: name: test_128bit_struct
; CHECK: $x0 = COPY
; CHECK: $x1 = COPY
; CHECK: $x2 = COPY
; CHECK: BL @take_128bit_struct
define void @test_128bit_struct([2 x i64]* %ptr) {
  %struct = load [2 x i64], [2 x i64]* %ptr
  call void @take_128bit_struct([2 x i64]* null, [2 x i64] %struct)
  ret void
}

; CHECK-LABEL: name: take_128bit_struct
; CHECK: {{%.*}}:_(p0) = COPY $x0
; CHECK: {{%.*}}:_(s64) = COPY $x1
; CHECK: {{%.*}}:_(s64) = COPY $x2
define void @take_128bit_struct([2 x i64]* %ptr, [2 x i64] %in) {
  store [2 x i64] %in, [2 x i64]* %ptr
  ret void
}

; CHECK-LABEL: name: test_split_struct
; CHECK: [[LD1:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load (s64) from %ir.ptr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s64) = G_LOAD %3(p0) :: (load (s64) from %ir.ptr + 8)

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFF]](s64)
; CHECK: G_STORE [[LD1]](s64), [[ADDR]](p0) :: (store (s64) into stack, align 1)

; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[CST]]
; CHECK: G_STORE [[LD2]](s64), [[ADDR]](p0) :: (store (s64) into stack + 8, align 1)
define void @test_split_struct([2 x i64]* %ptr) {
  %struct = load [2 x i64], [2 x i64]* %ptr
  call void @take_split_struct([2 x i64]* null, i64 1, i64 2, i64 3,
                               i64 4, i64 5, i64 6,
                               [2 x i64] %struct)
  ret void
}

; CHECK-LABEL: name: take_split_struct
; CHECK: fixedStack:
; CHECK-DAG:   - { id: [[LO_FRAME:[0-9]+]], type: default, offset: 0, size: 8
; CHECK-DAG:   - { id: [[HI_FRAME:[0-9]+]], type: default, offset: 8, size: 8

; CHECK: [[LOPTR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[LO_FRAME]]
; CHECK: [[LO:%[0-9]+]]:_(s64) = G_LOAD [[LOPTR]](p0) :: (invariant load (s64) from %fixed-stack.[[LO_FRAME]], align 16)

; CHECK: [[HIPTR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[HI_FRAME]]
; CHECK: [[HI:%[0-9]+]]:_(s64) = G_LOAD [[HIPTR]](p0) :: (invariant load (s64) from %fixed-stack.[[HI_FRAME]])
define void @take_split_struct([2 x i64]* %ptr, i64, i64, i64,
                               i64, i64, i64,
                               [2 x i64] %in) {
  store [2 x i64] %in, [2 x i64]* %ptr
  ret void
}
