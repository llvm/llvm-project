; RUN: llc -mtriple=aarch64-linux-gnu -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-LABEL: name: test_trivial_call
; CHECK: ADJCALLSTACKDOWN 0, 0, implicit-def %sp, implicit %sp
; CHECK: BL @trivial_callee, csr_aarch64_aapcs, implicit-def %lr
; CHECK: ADJCALLSTACKUP 0, 0, implicit-def %sp, implicit %sp
declare void @trivial_callee()
define void @test_trivial_call() {
  call void @trivial_callee()
  ret void
}

; CHECK-LABEL: name: test_simple_return
; CHECK: BL @simple_return_callee, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit-def %x0
; CHECK: [[RES:%[0-9]+]](s64) = COPY %x0
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
declare i64 @simple_return_callee()
define i64 @test_simple_return() {
  %res = call i64 @simple_return_callee()
  ret i64 %res
}

; CHECK-LABEL: name: test_simple_arg
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w0
; CHECK: %w0 = COPY [[IN]]
; CHECK: BL @simple_arg_callee, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0
; CHECK: RET_ReallyLR
declare void @simple_arg_callee(i32 %in)
define void @test_simple_arg(i32 %in) {
  call void @simple_arg_callee(i32 %in)
  ret void
}

; CHECK-LABEL: name: test_indirect_call
; CHECK: registers:
; Make sure the register feeding the indirect call is properly constrained.
; CHECK: - { id: [[FUNC:[0-9]+]], class: gpr64, preferred-register: '' }
; CHECK: %[[FUNC]](p0) = COPY %x0
; CHECK: BLR %[[FUNC]](p0), csr_aarch64_aapcs, implicit-def %lr, implicit %sp
; CHECK: RET_ReallyLR
define void @test_indirect_call(void()* %func) {
  call void %func()
  ret void
}

; CHECK-LABEL: name: test_multiple_args
; CHECK: [[IN:%[0-9]+]](s64) = COPY %x0
; CHECK: [[ANSWER:%[0-9]+]](s32) = G_CONSTANT i32 42
; CHECK: %w0 = COPY [[ANSWER]]
; CHECK: %x1 = COPY [[IN]]
; CHECK: BL @multiple_args_callee, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0, implicit %x1
; CHECK: RET_ReallyLR
declare void @multiple_args_callee(i32, i64)
define void @test_multiple_args(i64 %in) {
  call void @multiple_args_callee(i32 42, i64 %in)
  ret void
}


; CHECK-LABEL: name: test_struct_formal
; CHECK: [[DBL:%[0-9]+]](s64) = COPY %d0
; CHECK: [[I64:%[0-9]+]](s64) = COPY %x0
; CHECK: [[I8:%[0-9]+]](s8) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2

; CHECK: [[UNDEF:%[0-9]+]](s192) = IMPLICIT_DEF
; CHECK: [[ARG0:%[0-9]+]](s192) = G_INSERT [[UNDEF]], [[DBL]](s64), 0
; CHECK: [[ARG1:%[0-9]+]](s192) = G_INSERT [[ARG0]], [[I64]](s64), 64
; CHECK: [[ARG2:%[0-9]+]](s192) = G_INSERT [[ARG1]], [[I8]](s8), 128
; CHECK: [[ARG:%[0-9]+]](s192) = COPY [[ARG2]]

; CHECK: G_STORE [[ARG]](s192), [[ADDR]](p0)
; CHECK: RET_ReallyLR
define void @test_struct_formal({double, i64, i8} %in, {double, i64, i8}* %addr) {
  store {double, i64, i8} %in, {double, i64, i8}* %addr
  ret void
}


; CHECK-LABEL: name: test_struct_return
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]](s192) = G_LOAD [[ADDR]](p0)

; CHECK: [[DBL:%[0-9]+]](s64) = G_EXTRACT [[VAL]](s192), 0
; CHECK: [[I64:%[0-9]+]](s64) = G_EXTRACT [[VAL]](s192), 64
; CHECK: [[I32:%[0-9]+]](s32) = G_EXTRACT [[VAL]](s192), 128

; CHECK: %d0 = COPY [[DBL]](s64)
; CHECK: %x0 = COPY [[I64]](s64)
; CHECK: %w1 = COPY [[I32]](s32)
; CHECK: RET_ReallyLR implicit %d0, implicit %x0, implicit %w1
define {double, i64, i32} @test_struct_return({double, i64, i32}* %addr) {
  %val = load {double, i64, i32}, {double, i64, i32}* %addr
  ret {double, i64, i32} %val
}

; CHECK-LABEL: name: test_arr_call
; CHECK: hasCalls: true
; CHECK: [[ARG:%[0-9]+]](s256) = G_LOAD

; CHECK: [[E0:%[0-9]+]](s64) = G_EXTRACT [[ARG]](s256), 0
; CHECK: [[E1:%[0-9]+]](s64) = G_EXTRACT [[ARG]](s256), 64
; CHECK: [[E2:%[0-9]+]](s64) = G_EXTRACT [[ARG]](s256), 128
; CHECK: [[E3:%[0-9]+]](s64) = G_EXTRACT [[ARG]](s256), 192

; CHECK: %x0 = COPY [[E0]](s64)
; CHECK: %x1 = COPY [[E1]](s64)
; CHECK: %x2 = COPY [[E2]](s64)
; CHECK: %x3 = COPY [[E3]](s64)
; CHECK: BL @arr_callee, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %x0, implicit %x1, implicit %x2, implicit %x3, implicit-def %x0, implicit-def %x1, implicit-def %x2, implicit-def %x3
; CHECK: [[E0:%[0-9]+]](s64) = COPY %x0
; CHECK: [[E1:%[0-9]+]](s64) = COPY %x1
; CHECK: [[E2:%[0-9]+]](s64) = COPY %x2
; CHECK: [[E3:%[0-9]+]](s64) = COPY %x3
; CHECK: [[RES:%[0-9]+]](s256) = G_SEQUENCE [[E0]](s64), 0, [[E1]](s64), 64, [[E2]](s64), 128, [[E3]](s64), 192
; CHECK: G_EXTRACT [[RES]](s256), 64
declare [4 x i64] @arr_callee([4 x i64])
define i64 @test_arr_call([4 x i64]* %addr) {
  %arg = load [4 x i64], [4 x i64]* %addr
  %res = call [4 x i64] @arr_callee([4 x i64] %arg)
  %val = extractvalue [4 x i64] %res, 1
  ret i64 %val
}


; CHECK-LABEL: name: test_abi_exts_call
; CHECK: [[VAL:%[0-9]+]](s8) = G_LOAD
; CHECK: %w0 = COPY [[VAL]]
; CHECK: BL @take_char, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0
; CHECK: [[SVAL:%[0-9]+]](s32) = G_SEXT [[VAL]](s8)
; CHECK: %w0 = COPY [[SVAL]](s32)
; CHECK: BL @take_char, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0
; CHECK: [[ZVAL:%[0-9]+]](s32) = G_ZEXT [[VAL]](s8)
; CHECK: %w0 = COPY [[ZVAL]](s32)
; CHECK: BL @take_char, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0
declare void @take_char(i8)
define void @test_abi_exts_call(i8* %addr) {
  %val = load i8, i8* %addr
  call void @take_char(i8 %val)
  call void @take_char(i8 signext %val)
  call void @take_char(i8 zeroext %val)
  ret void
}

; CHECK-LABEL: name: test_abi_sext_ret
; CHECK: [[VAL:%[0-9]+]](s8) = G_LOAD
; CHECK: [[SVAL:%[0-9]+]](s32) = G_SEXT [[VAL]](s8)
; CHECK: %w0 = COPY [[SVAL]](s32)
; CHECK: RET_ReallyLR implicit %w0
define signext i8 @test_abi_sext_ret(i8* %addr) {
  %val = load i8, i8* %addr
  ret i8 %val
}

; CHECK-LABEL: name: test_abi_zext_ret
; CHECK: [[VAL:%[0-9]+]](s8) = G_LOAD
; CHECK: [[SVAL:%[0-9]+]](s32) = G_ZEXT [[VAL]](s8)
; CHECK: %w0 = COPY [[SVAL]](s32)
; CHECK: RET_ReallyLR implicit %w0
define zeroext i8 @test_abi_zext_ret(i8* %addr) {
  %val = load i8, i8* %addr
  ret i8 %val
}

; CHECK-LABEL: name: test_stack_slots
; CHECK: fixedStack:
; CHECK-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 8,
; CHECK-DAG:  - { id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 8,
; CHECK-DAG:  - { id: [[STACK16:[0-9]+]], type: default, offset: 16, size: 8,
; CHECK: [[LHS_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; CHECK: [[LHS:%[0-9]+]](s64) = G_LOAD [[LHS_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK0]], align 0)
; CHECK: [[RHS_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; CHECK: [[RHS:%[0-9]+]](s64) = G_LOAD [[RHS_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK8]], align 0)
; CHECK: [[ADDR_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK16]]
; CHECK: [[ADDR:%[0-9]+]](p0) = G_LOAD [[ADDR_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK16]], align 0)
; CHECK: [[SUM:%[0-9]+]](s64) = G_ADD [[LHS]], [[RHS]]
; CHECK: G_STORE [[SUM]](s64), [[ADDR]](p0)
define void @test_stack_slots([8 x i64], i64 %lhs, i64 %rhs, i64* %addr) {
  %sum = add i64 %lhs, %rhs
  store i64 %sum, i64* %addr
  ret void
}

; CHECK-LABEL: name: test_call_stack
; CHECK: [[C42:%[0-9]+]](s64) = G_CONSTANT i64 42
; CHECK: [[C12:%[0-9]+]](s64) = G_CONSTANT i64 12
; CHECK: [[PTR:%[0-9]+]](p0) = G_CONSTANT i64 0
; CHECK: ADJCALLSTACKDOWN 24, 0, implicit-def %sp, implicit %sp
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[C42_OFFS:%[0-9]+]](s64) = G_CONSTANT i64 0
; CHECK: [[C42_LOC:%[0-9]+]](p0) = G_GEP [[SP]], [[C42_OFFS]](s64)
; CHECK: G_STORE [[C42]](s64), [[C42_LOC]](p0) :: (store 8 into stack, align 0)
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[C12_OFFS:%[0-9]+]](s64) = G_CONSTANT i64 8
; CHECK: [[C12_LOC:%[0-9]+]](p0) = G_GEP [[SP]], [[C12_OFFS]](s64)
; CHECK: G_STORE [[C12]](s64), [[C12_LOC]](p0) :: (store 8 into stack + 8, align 0)
; CHECK: [[SP:%[0-9]+]](p0) = COPY %sp
; CHECK: [[PTR_OFFS:%[0-9]+]](s64) = G_CONSTANT i64 16
; CHECK: [[PTR_LOC:%[0-9]+]](p0) = G_GEP [[SP]], [[PTR_OFFS]](s64)
; CHECK: G_STORE [[PTR]](p0), [[PTR_LOC]](p0) :: (store 8 into stack + 16, align 0)
; CHECK: BL @test_stack_slots
; CHECK: ADJCALLSTACKUP 24, 0, implicit-def %sp, implicit %sp
define void @test_call_stack() {
  call void @test_stack_slots([8 x i64] undef, i64 42, i64 12, i64* null)
  ret void
}

; CHECK-LABEL: name: test_mem_i1
; CHECK: fixedStack:
; CHECK-NEXT: - { id: [[SLOT:[0-9]+]], type: default, offset: 0, size: 1, alignment: 16, isImmutable: true,
; CHECK: [[ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[SLOT]]
; CHECK: {{%[0-9]+}}(s1) = G_LOAD [[ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[SLOT]], align 0)
define void @test_mem_i1([8 x i64], i1 %in) {
  ret void
}
