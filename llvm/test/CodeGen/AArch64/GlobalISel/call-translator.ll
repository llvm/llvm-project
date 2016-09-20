; RUN: llc -mtriple=aarch64-linux-gnu -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-LABEL: name: test_trivial_call
; CHECK: BL @trivial_callee, csr_aarch64_aapcs, implicit-def %lr
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
; CHECK: [[FUNC:%[0-9]+]](p0) = COPY %x0
; CHECK: BLR [[FUNC]](p0), csr_aarch64_aapcs, implicit-def %lr, implicit %sp
; CHECK: RET_ReallyLR
define void @test_indirect_call(void()* %func) {
  call void %func()
  ret void
}

; CHECK-LABEL: name: test_multiple_args
; CHECK: [[IN:%[0-9]+]](s64) = COPY %x0
; CHECK: [[ANSWER:%[0-9]+]](s32) = G_CONSTANT 42
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
; CHECK: [[ARG:%[0-9]+]](s192) = G_SEQUENCE [[DBL]](s64), 0, [[I64]](s64), 64, [[I8]](s8), 128
; CHECK: G_STORE [[ARG]](s192), [[ADDR]](p0)
; CHECK: RET_ReallyLR
define void @test_struct_formal({double, i64, i8} %in, {double, i64, i8}* %addr) {
  store {double, i64, i8} %in, {double, i64, i8}* %addr
  ret void
}


; CHECK-LABEL: name: test_struct_return
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]](s192) = G_LOAD [[ADDR]](p0)
; CHECK: [[DBL:%[0-9]+]](s64), [[I64:%[0-9]+]](s64), [[I32:%[0-9]+]](s32) = G_EXTRACT [[VAL]](s192), 0, 64, 128
; CHECK: %d0 = COPY [[DBL]](s64)
; CHECK: %x0 = COPY [[I64]](s64)
; CHECK: %w1 = COPY [[I32]](s32)
; CHECK: RET_ReallyLR implicit %d0, implicit %x0, implicit %w1
define {double, i64, i32} @test_struct_return({double, i64, i32}* %addr) {
  %val = load {double, i64, i32}, {double, i64, i32}* %addr
  ret {double, i64, i32} %val
}

; CHECK-LABEL: name: test_arr_call
; CHECK: [[ARG:%[0-9]+]](s256) = G_LOAD
; CHECK: [[E0:%[0-9]+]](s64), [[E1:%[0-9]+]](s64), [[E2:%[0-9]+]](s64), [[E3:%[0-9]+]](s64) = G_EXTRACT [[ARG]](s256), 0, 64, 128, 192
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
