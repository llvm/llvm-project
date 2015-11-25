; RUN: llc -O0 -mcpu=e500mc < %s | FileCheck %s
; Check if e500 generates code with mfocrf insn.

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-linux-gnu"

define internal i32 @func_49(i64 %p_50, i16 zeroext %p_51, i8* %p_52, i32 %p_53) {
; CHECK-LABEL: @func_49
; CHECK-NOT: mfocrf

  %1 = load i64, i64* undef, align 8
  %2 = load i64, i64* undef, align 8
  %3 = icmp sge i32 undef, undef
  %4 = zext i1 %3 to i32
  %5 = sext i32 %4 to i64
  %6 = icmp slt i64 %2, %5
  %7 = zext i1 %6 to i32
  %8 = call i64 @safe_sub_func_int64_t_s_s(i64 -6372137293439783564, i64 undef)
  %9 = icmp slt i32 %7, undef
  %10 = zext i1 %9 to i32
  %11 = sext i32 %10 to i64
  %12 = icmp sle i64 %1, %11
  %13 = zext i1 %12 to i32
  %14 = call i32 @safe_add_func_int32_t_s_s(i32 undef, i32 %13)
  ret i32 undef
}

declare i32 @safe_add_func_int32_t_s_s(i32, i32)

declare i64 @safe_sub_func_int64_t_s_s(i64, i64)
