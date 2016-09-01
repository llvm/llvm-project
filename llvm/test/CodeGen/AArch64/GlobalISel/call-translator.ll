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
; CHECK: [[RES:%[0-9]+]](64) = G_TYPE s64 %x0
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
declare i64 @simple_return_callee()
define i64 @test_simple_return() {
  %res = call i64 @simple_return_callee()
  ret i64 %res
}

; CHECK-LABEL: name: test_simple_arg
; CHECK: [[IN:%[0-9]+]](32) = G_TYPE s32 %w0
; CHECK: %w0 = COPY [[IN]]
; CHECK: BL @simple_arg_callee, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0
; CHECK: RET_ReallyLR
declare void @simple_arg_callee(i32 %in)
define void @test_simple_arg(i32 %in) {
  call void @simple_arg_callee(i32 %in)
  ret void
}

; CHECK-LABEL: name: test_indirect_call
; CHECK: [[FUNC:%[0-9]+]](64) = G_TYPE p0 %x0
; CHECK: BLR [[FUNC]], csr_aarch64_aapcs, implicit-def %lr, implicit %sp
; CHECK: RET_ReallyLR
define void @test_indirect_call(void()* %func) {
  call void %func()
  ret void
}

; CHECK-LABEL: name: test_multiple_args
; CHECK: [[IN:%[0-9]+]](64) = G_TYPE s64 %x0
; CHECK: [[ANSWER:%[0-9]+]](32) = G_CONSTANT s32 42
; CHECK: %w0 = COPY [[ANSWER]]
; CHECK: %x1 = COPY [[IN]]
; CHECK: BL @multiple_args_callee, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0, implicit %x1
; CHECK: RET_ReallyLR
declare void @multiple_args_callee(i32, i64)
define void @test_multiple_args(i64 %in) {
  call void @multiple_args_callee(i32 42, i64 %in)
  ret void
}
