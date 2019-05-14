; RUN: llc -mtriple=aarch64-linux-gnu -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-LABEL: name: test_trivial_call
; CHECK: ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
; CHECK: BL @trivial_callee, csr_aarch64_aapcs, implicit-def $lr
; CHECK: ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
declare void @trivial_callee()
define void @test_trivial_call() {
  call void @trivial_callee()
  ret void
}

; CHECK-LABEL: name: test_simple_return
; CHECK: BL @simple_return_callee, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit-def $x0
; CHECK: [[RES:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: $x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $x0
declare i64 @simple_return_callee()
define i64 @test_simple_return() {
  %res = call i64 @simple_return_callee()
  ret i64 %res
}

; CHECK-LABEL: name: test_simple_arg
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: $w0 = COPY [[IN]]
; CHECK: BL @simple_arg_callee, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0
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
; CHECK: %[[FUNC]]:gpr64(p0) = COPY $x0
; CHECK: BLR %[[FUNC]](p0), csr_aarch64_aapcs, implicit-def $lr, implicit $sp
; CHECK: RET_ReallyLR
define void @test_indirect_call(void()* %func) {
  call void %func()
  ret void
}

; CHECK-LABEL: name: test_multiple_args
; CHECK: [[IN:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[ANSWER:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
; CHECK: $w0 = COPY [[ANSWER]]
; CHECK: $x1 = COPY [[IN]]
; CHECK: BL @multiple_args_callee, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0, implicit $x1
; CHECK: RET_ReallyLR
declare void @multiple_args_callee(i32, i64)
define void @test_multiple_args(i64 %in) {
  call void @multiple_args_callee(i32 42, i64 %in)
  ret void
}


; CHECK-LABEL: name: test_struct_formal
; CHECK: [[DBL:%[0-9]+]]:_(s64) = COPY $d0
; CHECK: [[I64:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[I8_C:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[I8:%[0-9]+]]:_(s8) = G_TRUNC [[I8_C]]
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2

; CHECK: [[UNDEF:%[0-9]+]]:_(s192) = G_IMPLICIT_DEF
; CHECK: [[ARG0:%[0-9]+]]:_(s192) = G_INSERT [[UNDEF]], [[DBL]](s64), 0
; CHECK: [[ARG1:%[0-9]+]]:_(s192) = G_INSERT [[ARG0]], [[I64]](s64), 64
; CHECK: [[ARG2:%[0-9]+]]:_(s192) = G_INSERT [[ARG1]], [[I8]](s8), 128
; CHECK: [[ARG:%[0-9]+]]:_(s192) = COPY [[ARG2]]
; CHECK: [[EXTA0:%[0-9]+]]:_(s64) = G_EXTRACT [[ARG]](s192), 0
; CHECK: [[EXTA1:%[0-9]+]]:_(s64) = G_EXTRACT [[ARG]](s192), 64
; CHECK: [[EXTA2:%[0-9]+]]:_(s8) = G_EXTRACT [[ARG]](s192), 128
; CHECK: G_STORE [[EXTA0]](s64), [[ADDR]](p0) :: (store 8 into %ir.addr)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_GEP [[ADDR]], [[CST1]](s64)
; CHECK: G_STORE [[EXTA1]](s64), [[GEP1]](p0) :: (store 8 into %ir.addr + 8)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_GEP [[ADDR]], [[CST2]](s64)
; CHECK: G_STORE [[EXTA2]](s8), [[GEP2]](p0) :: (store 1 into %ir.addr + 16, align 8)
; CHECK: RET_ReallyLR
define void @test_struct_formal({double, i64, i8} %in, {double, i64, i8}* %addr) {
  store {double, i64, i8} %in, {double, i64, i8}* %addr
  ret void
}


; CHECK-LABEL: name: test_struct_return
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0

; CHECK: [[LD1:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (load 8 from %ir.addr)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_GEP [[ADDR]], [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s64) = G_LOAD [[GEP1]](p0) :: (load 8 from %ir.addr + 8)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_GEP [[ADDR]], [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s32) = G_LOAD [[GEP2]](p0) :: (load 4 from %ir.addr + 16, align 8)

; CHECK: $d0 = COPY [[LD1]](s64)
; CHECK: $x0 = COPY [[LD2]](s64)
; CHECK: $w1 = COPY [[LD3]](s32)
; CHECK: RET_ReallyLR implicit $d0, implicit $x0, implicit $w1
define {double, i64, i32} @test_struct_return({double, i64, i32}* %addr) {
  %val = load {double, i64, i32}, {double, i64, i32}* %addr
  ret {double, i64, i32} %val
}

; CHECK-LABEL: name: test_arr_call
; CHECK: hasCalls: true
; CHECK: %0:_(p0) = COPY $x0
; CHECK: [[LD1:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load 8 from %ir.addr)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_GEP %0, [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s64) = G_LOAD [[GEP1]](p0) :: (load 8 from %ir.addr + 8)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_GEP %0, [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s64) = G_LOAD [[GEP2]](p0) :: (load 8 from %ir.addr + 16)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 24
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_GEP %0, [[CST3]](s64)
; CHECK: [[LD4:%[0-9]+]]:_(s64) = G_LOAD [[GEP3]](p0) :: (load 8 from %ir.addr + 24)
; CHECK: [[IMPDEF:%[0-9]+]]:_(s256) = G_IMPLICIT_DEF
; CHECK: [[INS1:%[0-9]+]]:_(s256) = G_INSERT [[IMPDEF]], [[LD1]](s64), 0
; CHECK: [[INS2:%[0-9]+]]:_(s256) = G_INSERT [[INS1]], [[LD2]](s64), 64
; CHECK: [[INS3:%[0-9]+]]:_(s256) = G_INSERT [[INS2]], [[LD3]](s64), 128
; CHECK: [[ARG:%[0-9]+]]:_(s256) = G_INSERT [[INS3]], [[LD4]](s64), 192
; CHECK: [[E0:%[0-9]+]]:_(s64) = G_EXTRACT [[ARG]](s256), 0
; CHECK: [[E1:%[0-9]+]]:_(s64) = G_EXTRACT [[ARG]](s256), 64
; CHECK: [[E2:%[0-9]+]]:_(s64) = G_EXTRACT [[ARG]](s256), 128
; CHECK: [[E3:%[0-9]+]]:_(s64) = G_EXTRACT [[ARG]](s256), 192

; CHECK: $x0 = COPY [[E0]](s64)
; CHECK: $x1 = COPY [[E1]](s64)
; CHECK: $x2 = COPY [[E2]](s64)
; CHECK: $x3 = COPY [[E3]](s64)
; CHECK: BL @arr_callee, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $x0, implicit $x1, implicit $x2, implicit $x3, implicit-def $x0, implicit-def $x1, implicit-def $x2, implicit-def $x3
; CHECK: [[E0:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[E1:%[0-9]+]]:_(s64) = COPY $x1
; CHECK: [[E2:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: [[E3:%[0-9]+]]:_(s64) = COPY $x3
; CHECK: [[RES:%[0-9]+]]:_(s256) = G_MERGE_VALUES [[E0]](s64), [[E1]](s64), [[E2]](s64), [[E3]](s64)
; CHECK: G_EXTRACT [[RES]](s256), 64
declare [4 x i64] @arr_callee([4 x i64])
define i64 @test_arr_call([4 x i64]* %addr) {
  %arg = load [4 x i64], [4 x i64]* %addr
  %res = call [4 x i64] @arr_callee([4 x i64] %arg)
  %val = extractvalue [4 x i64] %res, 1
  ret i64 %val
}


; CHECK-LABEL: name: test_abi_exts_call
; CHECK: [[VAL:%[0-9]+]]:_(s8) = G_LOAD
; CHECK: [[VAL_TMP:%[0-9]+]]:_(s32) = G_ANYEXT [[VAL]]
; CHECK: $w0 = COPY [[VAL_TMP]]
; CHECK: BL @take_char, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0
; CHECK: [[SVAL:%[0-9]+]]:_(s32) = G_SEXT [[VAL]](s8)
; CHECK: $w0 = COPY [[SVAL]](s32)
; CHECK: BL @take_char, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0
; CHECK: [[ZVAL:%[0-9]+]]:_(s32) = G_ZEXT [[VAL]](s8)
; CHECK: $w0 = COPY [[ZVAL]](s32)
; CHECK: BL @take_char, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0
declare void @take_char(i8)
define void @test_abi_exts_call(i8* %addr) {
  %val = load i8, i8* %addr
  call void @take_char(i8 %val)
  call void @take_char(i8 signext %val)
  call void @take_char(i8 zeroext %val)
  ret void
}

; CHECK-LABEL: name: test_abi_sext_ret
; CHECK: [[VAL:%[0-9]+]]:_(s8) = G_LOAD
; CHECK: [[SVAL:%[0-9]+]]:_(s32) = G_SEXT [[VAL]](s8)
; CHECK: $w0 = COPY [[SVAL]](s32)
; CHECK: RET_ReallyLR implicit $w0
define signext i8 @test_abi_sext_ret(i8* %addr) {
  %val = load i8, i8* %addr
  ret i8 %val
}

; CHECK-LABEL: name: test_abi_zext_ret
; CHECK: [[VAL:%[0-9]+]]:_(s8) = G_LOAD
; CHECK: [[SVAL:%[0-9]+]]:_(s32) = G_ZEXT [[VAL]](s8)
; CHECK: $w0 = COPY [[SVAL]](s32)
; CHECK: RET_ReallyLR implicit $w0
define zeroext i8 @test_abi_zext_ret(i8* %addr) {
  %val = load i8, i8* %addr
  ret i8 %val
}

; CHECK-LABEL: name: test_stack_slots
; CHECK: fixedStack:
; CHECK-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 8,
; CHECK-DAG:  - { id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 8,
; CHECK-DAG:  - { id: [[STACK16:[0-9]+]], type: default, offset: 16, size: 8,
; CHECK: [[LHS_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; CHECK: [[LHS:%[0-9]+]]:_(s64) = G_LOAD [[LHS_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK0]], align 1)
; CHECK: [[RHS_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; CHECK: [[RHS:%[0-9]+]]:_(s64) = G_LOAD [[RHS_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK8]], align 1)
; CHECK: [[ADDR_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK16]]
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_LOAD [[ADDR_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK16]], align 1)
; CHECK: [[SUM:%[0-9]+]]:_(s64) = G_ADD [[LHS]], [[RHS]]
; CHECK: G_STORE [[SUM]](s64), [[ADDR]](p0)
define void @test_stack_slots([8 x i64], i64 %lhs, i64 %rhs, i64* %addr) {
  %sum = add i64 %lhs, %rhs
  store i64 %sum, i64* %addr
  ret void
}

; CHECK-LABEL: name: test_call_stack
; CHECK: [[C42:%[0-9]+]]:_(s64) = G_CONSTANT i64 42
; CHECK: [[C12:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[ZERO:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_INTTOPTR [[ZERO]]
; CHECK: ADJCALLSTACKDOWN 24, 0, implicit-def $sp, implicit $sp
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[C42_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[C42_LOC:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[C42_OFFS]](s64)
; CHECK: G_STORE [[C42]](s64), [[C42_LOC]](p0) :: (store 8 into stack, align 1)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[C12_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[C12_LOC:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[C12_OFFS]](s64)
; CHECK: G_STORE [[C12]](s64), [[C12_LOC]](p0) :: (store 8 into stack + 8, align 1)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[PTR_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[PTR_LOC:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[PTR_OFFS]](s64)
; CHECK: G_STORE [[PTR]](p0), [[PTR_LOC]](p0) :: (store 8 into stack + 16, align 1)
; CHECK: BL @test_stack_slots
; CHECK: ADJCALLSTACKUP 24, 0, implicit-def $sp, implicit $sp
define void @test_call_stack() {
  call void @test_stack_slots([8 x i64] undef, i64 42, i64 12, i64* null)
  ret void
}

; CHECK-LABEL: name: test_mem_i1
; CHECK: fixedStack:
; CHECK-NEXT: - { id: [[SLOT:[0-9]+]], type: default, offset: 0, size: 1, alignment: 16, stack-id: 0,
; CHECK-NEXT: isImmutable: true,
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[SLOT]]
; CHECK: {{%[0-9]+}}:_(s1) = G_LOAD [[ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[SLOT]])
define void @test_mem_i1([8 x i64], i1 %in) {
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
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[LO:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load 8 from %ir.ptr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_GEP [[ADDR]], [[CST]](s64)
; CHECK: [[HI:%[0-9]+]]:_(s64) = G_LOAD [[GEP]](p0) :: (load 8 from %ir.ptr + 8)

; CHECK: [[IMPDEF:%[0-9]+]]:_(s128) = G_IMPLICIT_DEF
; CHECK: [[INS1:%[0-9]+]]:_(s128) = G_INSERT [[IMPDEF]], [[LO]](s64), 0
; CHECK: [[INS2:%[0-9]+]]:_(s128) = G_INSERT [[INS1]], [[HI]](s64), 64
; CHECK: [[EXTLO:%[0-9]+]]:_(s64) = G_EXTRACT [[INS2]](s128), 0
; CHECK: [[EXTHI:%[0-9]+]]:_(s64) = G_EXTRACT [[INS2]](s128), 64

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[CST2]](s64)
; CHECK: G_STORE [[EXTLO]](s64), [[GEP2]](p0) :: (store 8 into stack, align 1)
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[CST3:%[0-9]+]]:_(s64) = COPY [[CST]]
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_GEP [[SP]], [[CST3]](s64)
; CHECK: G_STORE [[EXTHI]](s64), [[GEP3]](p0) :: (store 8 into stack + 8, align 1)
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
