; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios9.0"

; CHECK-LABEL: name: test_varargs
; CHECK: [[ANSWER:%[0-9]+]](s32) = G_CONSTANT i32 42
; CHECK: [[D_ONE:%[0-9]+]](s64) = G_FCONSTANT double 1.000000e+00
; CHECK: [[TWELVE:%[0-9]+]](s64) = G_CONSTANT i64 12
; CHECK: [[THREE:%[0-9]+]](s8) = G_CONSTANT i8 3
; CHECK: [[ONE:%[0-9]+]](s16) = G_CONSTANT i16 1
; CHECK: [[FOUR:%[0-9]+]](s32) = G_CONSTANT i32 4
; CHECK: [[F_ONE:%[0-9]+]](s32) = G_FCONSTANT float 1.000000e+00
; CHECK: [[TWO:%[0-9]+]](s64) = G_FCONSTANT double 2.000000e+00

; CHECK: %w0 = COPY [[ANSWER]]
; CHECK: %d0 = COPY [[D_ONE]]
; CHECK: %x1 = COPY [[TWELVE]]
; CHECK: G_STORE [[THREE]](s8), {{%[0-9]+}}(p0) :: (store 1 into stack, align 0)
; CHECK: G_STORE [[ONE]](s16), {{%[0-9]+}}(p0) :: (store 2 into stack + 8, align 0)
; CHECK: G_STORE [[FOUR]](s32), {{%[0-9]+}}(p0) :: (store 4 into stack + 16, align 0)
; CHECK: G_STORE [[F_ONE]](s32), {{%[0-9]+}}(p0) :: (store 4 into stack + 24, align 0)
; CHECK: G_STORE [[TWO]](s64), {{%[0-9]+}}(p0) :: (store 8 into stack + 32, align 0)
declare void @varargs(i32, double, i64, ...)
define void @test_varargs() {
  call void(i32, double, i64, ...) @varargs(i32 42, double 1.0, i64 12, i8 3, i16 1, i32 4, float 1.0, double 2.0)
  ret void
}
