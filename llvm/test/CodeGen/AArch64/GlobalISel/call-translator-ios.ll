; RUN: llc -mtriple=aarch64-apple-ios -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s


; CHECK-LABEL: name: test_stack_slots
; CHECK: fixedStack:
; CHECK-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 1,
; CHECK-DAG:  - { id: [[STACK8:[0-9]+]], type: default, offset: 1, size: 1,
; CHECK: [[LHS_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; CHECK: [[LHS:%[0-9]+]]:_(s8) = G_LOAD [[LHS_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK0]])
; CHECK: [[RHS_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; CHECK: [[RHS:%[0-9]+]]:_(s8) = G_LOAD [[RHS_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK8]])
; CHECK: [[SUM:%[0-9]+]]:_(s8) = G_ADD [[LHS]], [[RHS]]
; CHECK: [[SUM32:%[0-9]+]]:_(s32) = G_SEXT [[SUM]](s8)
; CHECK: $w0 = COPY [[SUM32]](s32)
define signext i8 @test_stack_slots([8 x i64], i8 signext %lhs, i8 signext %rhs) {
  %sum = add i8 %lhs, %rhs
  ret i8 %sum
}

; CHECK-LABEL: name: test_call_stack
; CHECK: [[C42:%[0-9]+]]:_(s8) = G_CONSTANT i8 42
; CHECK: [[C12:%[0-9]+]]:_(s8) = G_CONSTANT i8 12
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[C42_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[C42_LOC:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[C42_OFFS]](s64)
; CHECK: G_STORE [[C42]](s8), [[C42_LOC]](p0) :: (store 1 into stack)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[C12_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: [[C12_LOC:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[C12_OFFS]](s64)
; CHECK: G_STORE [[C12]](s8), [[C12_LOC]](p0) :: (store 1 into stack + 1)
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
; CHECK: [[LD1:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load 8 from %ir.ptr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_GEP %0, [[CST]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s64) = G_LOAD %3(p0) :: (load 8 from %ir.ptr + 8)
; CHECK: [[IMPDEF:%[0-9]+]]:_(s128) = G_IMPLICIT_DEF
; CHECK: [[INS1:%[0-9]+]]:_(s128) = G_INSERT [[IMPDEF]], [[LD1]](s64), 0
; CHECK: [[INS2:%[0-9]+]]:_(s128) = G_INSERT [[INS1]], [[LD2]](s64), 64
; CHECK:  [[EXT1:%[0-9]+]]:_(s64) = G_EXTRACT [[INS2]](s128), 0
; CHECK: [[EXT2:%[0-9]+]]:_(s64) = G_EXTRACT [[INS2]](s128), 64

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[OFF]](s64)
; CHECK: G_STORE [[EXT1]](s64), [[ADDR]](p0) :: (store 8 into stack, align 1)

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFF:%[0-9]+]]:_(s64) = COPY [[CST]]
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[OFF]]
; CHECK: G_STORE [[EXT2]](s64), [[ADDR]](p0) :: (store 8 into stack + 8, align 1)
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
; CHECK: [[LO:%[0-9]+]]:_(s64) = G_LOAD [[LOPTR]](p0) :: (invariant load 8 from %fixed-stack.[[LO_FRAME]], align 1)

; CHECK: [[HIPTR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[HI_FRAME]]
; CHECK: [[HI:%[0-9]+]]:_(s64) = G_LOAD [[HIPTR]](p0) :: (invariant load 8 from %fixed-stack.[[HI_FRAME]], align 1)
define void @take_split_struct([2 x i64]* %ptr, i64, i64, i64,
                               i64, i64, i64,
                               [2 x i64] %in) {
  store [2 x i64] %in, [2 x i64]* %ptr
  ret void
}
