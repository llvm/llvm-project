; RUN: llc -O0 -stop-after=irtranslator -global-isel -global-isel-abort=1 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; CHECK-LABEL: name: args_i32
; CHECK: %[[ARG0:[0-9]+]]:_(s32) = COPY $w0
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w1
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w2
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w3
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w4
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w5
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w6
; CHECK: %{{[0-9]+}}:_(s32) = COPY $w7
; CHECK: $w0 = COPY %[[ARG0]]

define i32 @args_i32(i32 %w0, i32 %w1, i32 %w2, i32 %w3,
                     i32 %w4, i32 %w5, i32 %w6, i32 %w7) {
  ret i32 %w0
}

; CHECK-LABEL: name: args_i64
; CHECK: %[[ARG0:[0-9]+]]:_(s64) = COPY $x0
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x1
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x2
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x3
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x4
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x5
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x6
; CHECK: %{{[0-9]+}}:_(s64) = COPY $x7
; CHECK: $x0 = COPY %[[ARG0]]
define i64 @args_i64(i64 %x0, i64 %x1, i64 %x2, i64 %x3,
                     i64 %x4, i64 %x5, i64 %x6, i64 %x7) {
  ret i64 %x0
}


; CHECK-LABEL: name: args_ptrs
; CHECK: %[[ARG0:[0-9]+]]:_(p0) = COPY $x0
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x1
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x2
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x3
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x4
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x5
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x6
; CHECK: %{{[0-9]+}}:_(p0) = COPY $x7
; CHECK: $x0 = COPY %[[ARG0]]
define i8* @args_ptrs(i8* %x0, i16* %x1, <2 x i8>* %x2, {i8, i16, i32}* %x3,
                      [3 x float]* %x4, double* %x5, i8* %x6, i8* %x7) {
  ret i8* %x0
}

; CHECK-LABEL: name: args_arr
; CHECK: %[[ARG0:[0-9]+]]:_(s64) = COPY $d0
; CHECK: $d0 = COPY %[[ARG0]]
define [1 x double] @args_arr([1 x double] %d0) {
  ret [1 x double] %d0
}

; CHECK-LABEL: name: test_varargs
; CHECK: [[ANSWER:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
; CHECK: [[D_ONE:%[0-9]+]]:_(s64) = G_FCONSTANT double 1.000000e+00
; CHECK: [[TWELVE:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[THREE:%[0-9]+]]:_(s8) = G_CONSTANT i8 3
; CHECK: [[ONE:%[0-9]+]]:_(s16) = G_CONSTANT i16 1
; CHECK: [[FOUR:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[F_ONE:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.000000e+00
; CHECK: [[TWO:%[0-9]+]]:_(s64) = G_FCONSTANT double 2.000000e+00

; CHECK: $w0 = COPY [[ANSWER]]
; CHECK: $d0 = COPY [[D_ONE]]
; CHECK: $x1 = COPY [[TWELVE]]
; CHECK: [[THREE_TMP:%[0-9]+]]:_(s32) = G_ANYEXT [[THREE]]
; CHECK: $w2 = COPY [[THREE_TMP]](s32)
; CHECK: [[ONE_TMP:%[0-9]+]]:_(s32) = G_ANYEXT [[ONE]]
; CHECK: $w3 = COPY [[ONE_TMP]](s32)
; CHECK: $w4 = COPY [[FOUR]](s32)
; CHECK: $s1 = COPY [[F_ONE]](s32)
; CHECK: $d2 = COPY [[TWO]](s64)
declare void @varargs(i32, double, i64, ...)
define void @test_varargs() {
  call void(i32, double, i64, ...) @varargs(i32 42, double 1.0, i64 12, i8 3, i16 1, i32 4, float 1.0, double 2.0)
  ret void
}

; signext/zeroext parameters on the stack: not part of any real ABI as far as I
; know, but ELF currently allocates 8 bytes for a signext parameter on the
; stack. The ADJCALLSTACK ops should reflect this, even if the difference is
; theoretical.
declare void @stack_ext_needed([8 x i64], i8 signext %in)
; CHECK-LABEL: name: test_stack_ext_needed
; CHECK: ADJCALLSTACKDOWN 8
; CHECK: BL @stack_ext_needed
; CHECK: ADJCALLSTACKUP 8
define void @test_stack_ext_needed() {
  call void @stack_ext_needed([8 x i64] undef, i8 signext 42)
  ret void
}

; Check that we can lower incoming i128 types into constituent s64 gprs.
; CHECK-LABEL: name: callee_s128
; CHECK: liveins: $x0, $x1, $x2, $x3, $x4
; CHECK: [[A1_P1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[A1_P2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK: [[A1_MERGE:%[0-9]+]]:_(s128) = G_MERGE_VALUES [[A1_P1]](s64), [[A1_P2]](s64)
; CHECK: [[A2_P1:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: [[A2_P2:%[0-9]+]]:_(s64) = COPY $x3
; CHECK: [[A2_MERGE:%[0-9]+]]:_(s128) = G_MERGE_VALUES [[A2_P1]](s64), [[A2_P2]](s64)
; CHECK: G_STORE [[A2_MERGE]](s128)
define void @callee_s128(i128 %a, i128 %b, i128 *%ptr) {
  store i128 %b, i128 *%ptr
  ret void
}

; Check we can lower outgoing s128 arguments into s64 gprs.
; CHECK-LABEL: name: caller_s128
; CHECK: [[PTR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[LARGE_VAL:%[0-9]+]]:_(s128) = G_LOAD [[PTR]](p0)
; CHECK: ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
; CHECK: [[A1_P1:%[0-9]+]]:_(s64), [[A1_P2:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[LARGE_VAL]](s128)
; CHECK: [[A2_P1:%[0-9]+]]:_(s64), [[A2_P2:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES %1(s128)
; CHECK: $x0 = COPY [[A1_P1]](s64)
; CHECK: $x1 = COPY [[A1_P2]](s64)
; CHECK: $x2 = COPY [[A2_P1]](s64)
; CHECK: $x3 = COPY [[A2_P2]](s64)
; CHECK: $x4 = COPY [[PTR]](p0)
; CHECK: BL @callee_s128, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $x0, implicit $x1, implicit $x2, implicit $x3, implicit $x4
; CHECK: ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
define void @caller_s128(i128 *%ptr) {
  %v = load i128, i128 *%ptr
  call void @callee_s128(i128 %v, i128 %v, i128 *%ptr)
  ret void
}
