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
define void @test_indirect_call(ptr %func) {
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

; CHECK: G_STORE [[DBL]](s64), [[ADDR]](p0) :: (store (s64) into %ir.addr)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST1]](s64)
; CHECK: G_STORE [[I64]](s64), [[GEP1]](p0) :: (store (s64) into %ir.addr + 8)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST2]](s64)
; CHECK: G_STORE [[I8]](s8), [[GEP2]](p0) :: (store (s8) into %ir.addr + 16, align 8)
; CHECK: RET_ReallyLR
define void @test_struct_formal({double, i64, i8} %in, ptr %addr) {
  store {double, i64, i8} %in, ptr %addr
  ret void
}


; CHECK-LABEL: name: test_struct_return
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0

; CHECK: [[LD1:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (load (s64) from %ir.addr)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s64) = G_LOAD [[GEP1]](p0) :: (load (s64) from %ir.addr + 8)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s32) = G_LOAD [[GEP2]](p0) :: (load (s32) from %ir.addr + 16, align 8)

; CHECK: $d0 = COPY [[LD1]](s64)
; CHECK: $x0 = COPY [[LD2]](s64)
; CHECK: $w1 = COPY [[LD3]](s32)
; CHECK: RET_ReallyLR implicit $d0, implicit $x0, implicit $w1
define {double, i64, i32} @test_struct_return(ptr %addr) {
  %val = load {double, i64, i32}, ptr %addr
  ret {double, i64, i32} %val
}

; CHECK-LABEL: name: test_arr_call
; CHECK: %0:_(p0) = COPY $x0
; CHECK: [[LD1:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load (s64) from %ir.addr)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s64) = G_LOAD [[GEP1]](p0) :: (load (s64) from %ir.addr + 8)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s64) = G_LOAD [[GEP2]](p0) :: (load (s64) from %ir.addr + 16)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 24
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: [[LD4:%[0-9]+]]:_(s64) = G_LOAD [[GEP3]](p0) :: (load (s64) from %ir.addr + 24)

; CHECK: $x0 = COPY [[LD1]](s64)
; CHECK: $x1 = COPY [[LD2]](s64)
; CHECK: $x2 = COPY [[LD3]](s64)
; CHECK: $x3 = COPY [[LD4]](s64)
; CHECK: BL @arr_callee, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $x0, implicit $x1, implicit $x2, implicit $x3, implicit-def $x0, implicit-def $x1, implicit-def $x2, implicit-def $x3
; CHECK: [[E0:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[E1:%[0-9]+]]:_(s64) = COPY $x1
; CHECK: [[E2:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: [[E3:%[0-9]+]]:_(s64) = COPY $x3
; CHECK: $x0 = COPY [[E1]]
declare [4 x i64] @arr_callee([4 x i64])
define i64 @test_arr_call(ptr %addr) {
  %arg = load [4 x i64], ptr %addr
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
define void @test_abi_exts_call(ptr %addr) {
  %val = load i8, ptr %addr
  call void @take_char(i8 %val)
  call void @take_char(i8 signext %val)
  call void @take_char(i8 zeroext %val)
  ret void
}

; CHECK-LABEL: name: test_zext_in_callee
; CHECK: bb.1 (%ir-block.0):
; CHECK:   liveins: $x0
; CHECK:   [[COPY:%[0-9]+]]:_(p0) = COPY $x0
; CHECK:   [[LOAD:%[0-9]+]]:_(s8) = G_LOAD [[COPY]](p0) :: (load (s8) from %ir.addr)
; CHECK:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
; CHECK:   [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[LOAD]](s8)
; CHECK:   $w0 = COPY [[ZEXT]](s32)
; CHECK:   BL @has_zext_param, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0
; CHECK:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
; CHECK:   RET_ReallyLR
declare void @has_zext_param(i8 zeroext)
define void @test_zext_in_callee(ptr %addr) {
  %val = load i8, ptr %addr
  call void @has_zext_param(i8 %val)
  ret void
}

; CHECK-LABEL: name: test_sext_in_callee
; CHECK: bb.1 (%ir-block.0):
; CHECK:   liveins: $x0
; CHECK:   [[COPY:%[0-9]+]]:_(p0) = COPY $x0
; CHECK:   [[LOAD:%[0-9]+]]:_(s8) = G_LOAD [[COPY]](p0) :: (load (s8) from %ir.addr)
; CHECK:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
; CHECK:   [[SEXT:%[0-9]+]]:_(s32) = G_SEXT [[LOAD]](s8)
; CHECK:   $w0 = COPY [[SEXT]](s32)
; CHECK:   BL @has_sext_param, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0
; CHECK:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
; CHECK:   RET_ReallyLR
declare void @has_sext_param(i8 signext)
define void @test_sext_in_callee(ptr %addr) {
  %val = load i8, ptr %addr
  call void @has_sext_param(i8 %val)
  ret void
}

; CHECK-LABEL: name: test_abi_sext_ret
; CHECK: [[VAL:%[0-9]+]]:_(s8) = G_LOAD
; CHECK: [[SVAL:%[0-9]+]]:_(s32) = G_SEXT [[VAL]](s8)
; CHECK: $w0 = COPY [[SVAL]](s32)
; CHECK: RET_ReallyLR implicit $w0
define signext i8 @test_abi_sext_ret(ptr %addr) {
  %val = load i8, ptr %addr
  ret i8 %val
}

; CHECK-LABEL: name: test_abi_zext_ret
; CHECK: [[VAL:%[0-9]+]]:_(s8) = G_LOAD
; CHECK: [[SVAL:%[0-9]+]]:_(s32) = G_ZEXT [[VAL]](s8)
; CHECK: $w0 = COPY [[SVAL]](s32)
; CHECK: RET_ReallyLR implicit $w0
define zeroext i8 @test_abi_zext_ret(ptr %addr) {
  %val = load i8, ptr %addr
  ret i8 %val
}

; CHECK-LABEL: name: test_stack_slots
; CHECK: fixedStack:
; CHECK-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 8,
; CHECK-DAG:  - { id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 8,
; CHECK-DAG:  - { id: [[STACK16:[0-9]+]], type: default, offset: 16, size: 8,
; CHECK: [[LHS_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; CHECK: [[LHS:%[0-9]+]]:_(s64) = G_LOAD [[LHS_ADDR]](p0) :: (invariant load (s64) from %fixed-stack.[[STACK0]], align 16)
; CHECK: [[RHS_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; CHECK: [[RHS:%[0-9]+]]:_(s64) = G_LOAD [[RHS_ADDR]](p0) :: (invariant load (s64) from %fixed-stack.[[STACK8]])
; CHECK: [[ADDR_ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[STACK16]]
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_LOAD [[ADDR_ADDR]](p0) :: (invariant load (p0) from %fixed-stack.[[STACK16]], align 16)
; CHECK: [[SUM:%[0-9]+]]:_(s64) = G_ADD [[LHS]], [[RHS]]
; CHECK: G_STORE [[SUM]](s64), [[ADDR]](p0)
define void @test_stack_slots([8 x i64], i64 %lhs, i64 %rhs, ptr %addr) {
  %sum = add i64 %lhs, %rhs
  store i64 %sum, ptr %addr
  ret void
}

; CHECK-LABEL: name: test_call_stack
; CHECK: [[C42:%[0-9]+]]:_(s64) = G_CONSTANT i64 42
; CHECK: [[C12:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_CONSTANT i64 0
; CHECK: ADJCALLSTACKDOWN 24, 0, implicit-def $sp, implicit $sp
; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[C42_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[C42_LOC:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[C42_OFFS]](s64)
; CHECK: G_STORE [[C42]](s64), [[C42_LOC]](p0) :: (store (s64) into stack, align 1)
; CHECK: [[C12_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[C12_LOC:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[C12_OFFS]](s64)
; CHECK: G_STORE [[C12]](s64), [[C12_LOC]](p0) :: (store (s64) into stack + 8, align 1)
; CHECK: [[PTR_OFFS:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[PTR_LOC:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[PTR_OFFS]](s64)
; CHECK: G_STORE [[PTR]](p0), [[PTR_LOC]](p0) :: (store (p0) into stack + 16, align 1)
; CHECK: BL @test_stack_slots
; CHECK: ADJCALLSTACKUP 24, 0, implicit-def $sp, implicit $sp
define void @test_call_stack() {
  call void @test_stack_slots([8 x i64] undef, i64 42, i64 12, ptr null)
  ret void
}

; CHECK-LABEL: name: test_mem_i1
; CHECK: fixedStack:
; CHECK-NEXT: - { id: [[SLOT:[0-9]+]], type: default, offset: 0, size: 1, alignment: 16, stack-id: default,
; CHECK-NEXT: isImmutable: true,
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.[[SLOT]]
; CHECK: [[LOAD:%[0-9]+]]:_(s32) = G_LOAD [[ADDR]](p0) :: (invariant load (s8) from %fixed-stack.[[SLOT]], align 16)
; CHECK: [[TRUNC8:%[0-9]+]]:_(s8) = G_TRUNC [[LOAD]]
; CHECK: [[TRUNCASSERT:%[0-9]+]]:_(s8) = G_ASSERT_ZEXT [[TRUNC8]], 1
; CHECK: {{%[0-9]+}}:_(s1) = G_TRUNC [[TRUNCASSERT]]

define void @test_mem_i1([8 x i64], i1 %in) {
  ret void
}

; CHECK-LABEL: name: test_128bit_struct
; CHECK: $x0 = COPY
; CHECK: $x1 = COPY
; CHECK: $x2 = COPY
; CHECK: BL @take_128bit_struct
define void @test_128bit_struct(ptr %ptr) {
  %struct = load [2 x i64], ptr %ptr
  call void @take_128bit_struct(ptr null, [2 x i64] %struct)
  ret void
}

; CHECK-LABEL: name: take_128bit_struct
; CHECK: {{%.*}}:_(p0) = COPY $x0
; CHECK: {{%.*}}:_(s64) = COPY $x1
; CHECK: {{%.*}}:_(s64) = COPY $x2
define void @take_128bit_struct(ptr %ptr, [2 x i64] %in) {
  store [2 x i64] %in, ptr %ptr
  ret void
}

; CHECK-LABEL: name: test_split_struct
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[LO:%[0-9]+]]:_(s64) = G_LOAD %0(p0) :: (load (s64) from %ir.ptr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: [[HI:%[0-9]+]]:_(s64) = G_LOAD [[GEP]](p0) :: (load (s64) from %ir.ptr + 8)

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[CST2]](s64)
; CHECK: G_STORE [[LO]](s64), [[GEP2]](p0) :: (store (s64) into stack, align 1)
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[CST]](s64)
; CHECK: G_STORE [[HI]](s64), [[GEP3]](p0) :: (store (s64) into stack + 8, align 1)
define void @test_split_struct(ptr %ptr) {
  %struct = load [2 x i64], ptr %ptr
  call void @take_split_struct(ptr null, i64 1, i64 2, i64 3,
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
define void @take_split_struct(ptr %ptr, i64, i64, i64,
                               i64, i64, i64,
                               [2 x i64] %in) {
  store [2 x i64] %in, ptr %ptr
  ret void
}

%size0type = type { }
declare %size0type @func.returns.size0.struct()

; CHECK-LABEL: name: call_returns_size0_struct
; CHECK: bb.1
; CHECK-NEXT: ADJCALLSTACKDOWN
; CHECK-NEXT: BL
; CHECK-NEXT: ADJCALLSTACKUP
; CHECK-NEXT: RET_ReallyLR
define void @call_returns_size0_struct() {
  ; FIXME: Why is this valid IR?
  %call = call %size0type @func.returns.size0.struct()
  ret void
}

declare [0 x i8] @func.returns.size0.array()

; CHECK-LABEL: name: call_returns_size0_array
; CHECK: bb.1
; CHECK-NEXT: ADJCALLSTACKDOWN
; CHECK-NEXT: BL
; CHECK-NEXT: ADJCALLSTACKUP
; CHECK-NEXT: RET_ReallyLR
define void @call_returns_size0_array() {
  ; FIXME: Why is this valid IR?
  %call = call [0 x i8] @func.returns.size0.array()
  ret void
}

declare [1 x %size0type] @func.returns.array.size0.struct()
; CHECK-LABEL: name: call_returns_array_size0_struct
; CHECK: bb.1
; CHECK-NEXT: ADJCALLSTACKDOWN
; CHECK-NEXT: BL
; CHECK-NEXT: ADJCALLSTACKUP
; CHECK-NEXT: RET_ReallyLR
define void @call_returns_array_size0_struct() {
  ; FIXME: Why is this valid IR?
  %call = call [1 x %size0type] @func.returns.array.size0.struct()
  ret void
}

; Test extends on return values.

; This should notice that the return value from has_zext_return is zero
; extended.
declare zeroext i16 @has_zext_return()
define i32 @test_zext_return_from_callee() {
  ; CHECK-LABEL: name: test_zext_return_from_callee
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   BL @has_zext_return, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit-def $w0
  ; CHECK-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:_(s32) = COPY $w0
  ; CHECK-NEXT:   [[ASSERT_ZEXT:%[0-9]+]]:_(s32) = G_ASSERT_ZEXT [[COPY]], 16
  ; CHECK-NEXT:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[ASSERT_ZEXT]](s32)
  ; CHECK-NEXT:   [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[TRUNC]](s16)
  ; CHECK-NEXT:   $w0 = COPY [[ZEXT]](s32)
  ; CHECK-NEXT:   RET_ReallyLR implicit $w0
  %val = call i16 @has_zext_return()
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; Same as above, but with zeroext explicitly on the call. Should produce the
; same codegen.
define i32 @test_zext_return_from_callee2() {
  ; CHECK-LABEL: name: test_zext_return_from_callee2
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   BL @has_zext_return, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit-def $w0
  ; CHECK-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:_(s32) = COPY $w0
  ; CHECK-NEXT:   [[ASSERT_ZEXT:%[0-9]+]]:_(s32) = G_ASSERT_ZEXT [[COPY]], 16
  ; CHECK-NEXT:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[ASSERT_ZEXT]](s32)
  ; CHECK-NEXT:   [[ZEXT:%[0-9]+]]:_(s32) = G_ZEXT [[TRUNC]](s16)
  ; CHECK-NEXT:   $w0 = COPY [[ZEXT]](s32)
  ; CHECK-NEXT:   RET_ReallyLR implicit $w0
  %val = call zeroext i16 @has_zext_return()
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; This should notice that the return value from has_sext_return is sign
; extended.
declare signext i16 @has_sext_return()
define i32 @test_sext_return_from_callee() {
  ; CHECK-LABEL: name: test_sext_return_from_callee
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   BL @has_sext_return, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit-def $w0
  ; CHECK-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:_(s32) = COPY $w0
  ; CHECK-NEXT:   [[ASSERT_SEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[COPY]], 16
  ; CHECK-NEXT:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[ASSERT_SEXT]](s32)
  ; CHECK-NEXT:   [[SEXT:%[0-9]+]]:_(s32) = G_SEXT [[TRUNC]](s16)
  ; CHECK-NEXT:   $w0 = COPY [[SEXT]](s32)
  ; CHECK-NEXT:   RET_ReallyLR implicit $w0
  %val = call i16 @has_sext_return()
  %ext = sext i16 %val to i32
  ret i32 %ext
}

; Same as above, but with signext explicitly on the call. Should produce the
; same codegen.
define i32 @test_sext_return_from_callee2() {
  ; CHECK-LABEL: name: test_sext_return_from_callee2
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   BL @has_sext_return, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit-def $w0
  ; CHECK-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:_(s32) = COPY $w0
  ; CHECK-NEXT:   [[ASSERT_SEXT:%[0-9]+]]:_(s32) = G_ASSERT_SEXT [[COPY]], 16
  ; CHECK-NEXT:   [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[ASSERT_SEXT]](s32)
  ; CHECK-NEXT:   [[SEXT:%[0-9]+]]:_(s32) = G_SEXT [[TRUNC]](s16)
  ; CHECK-NEXT:   $w0 = COPY [[SEXT]](s32)
  ; CHECK-NEXT:   RET_ReallyLR implicit $w0
  %val = call signext i16 @has_sext_return()
  %ext = sext i16 %val to i32
  ret i32 %ext
}
