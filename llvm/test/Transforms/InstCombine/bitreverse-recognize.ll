; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define zeroext i8 @f_u8(i8 zeroext %a) {
; CHECK-LABEL: @f_u8
; CHECK-NEXT: %[[A:.*]] = call i8 @llvm.bitreverse.i8(i8 %a)
; CHECK-NEXT: ret i8 %[[A]]
  %1 = shl i8 %a, 7
  %2 = shl i8 %a, 5
  %3 = and i8 %2, 64
  %4 = shl i8 %a, 3
  %5 = and i8 %4, 32
  %6 = shl i8 %a, 1
  %7 = and i8 %6, 16
  %8 = lshr i8 %a, 1
  %9 = and i8 %8, 8
  %10 = lshr i8 %a, 3
  %11 = and i8 %10, 4
  %12 = lshr i8 %a, 5
  %13 = and i8 %12, 2
  %14 = lshr i8 %a, 7
  %15 = or i8 %14, %1
  %16 = or i8 %15, %3
  %17 = or i8 %16, %5
  %18 = or i8 %17, %7
  %19 = or i8 %18, %9
  %20 = or i8 %19, %11
  %21 = or i8 %20, %13
  ret i8 %21
}

; The ANDs with 32 and 64 have been swapped here, so the sequence does not
; completely match a bitreverse.
define zeroext i8 @f_u8_fail(i8 zeroext %a) {
; CHECK-LABEL: @f_u8_fail
; CHECK-NOT: call
; CHECK: ret i8
  %1 = shl i8 %a, 7
  %2 = shl i8 %a, 5
  %3 = and i8 %2, 32
  %4 = shl i8 %a, 3
  %5 = and i8 %4, 64
  %6 = shl i8 %a, 1
  %7 = and i8 %6, 16
  %8 = lshr i8 %a, 1
  %9 = and i8 %8, 8
  %10 = lshr i8 %a, 3
  %11 = and i8 %10, 4
  %12 = lshr i8 %a, 5
  %13 = and i8 %12, 2
  %14 = lshr i8 %a, 7
  %15 = or i8 %14, %1
  %16 = or i8 %15, %3
  %17 = or i8 %16, %5
  %18 = or i8 %17, %7
  %19 = or i8 %18, %9
  %20 = or i8 %19, %11
  %21 = or i8 %20, %13
  ret i8 %21
}

define zeroext i16 @f_u16(i16 zeroext %a) {
; CHECK-LABEL: @f_u16
; CHECK-NEXT: %[[A:.*]] = call i16 @llvm.bitreverse.i16(i16 %a)
; CHECK-NEXT: ret i16 %[[A]]
  %1 = shl i16 %a, 15
  %2 = shl i16 %a, 13
  %3 = and i16 %2, 16384
  %4 = shl i16 %a, 11
  %5 = and i16 %4, 8192
  %6 = shl i16 %a, 9
  %7 = and i16 %6, 4096
  %8 = shl i16 %a, 7
  %9 = and i16 %8, 2048
  %10 = shl i16 %a, 5
  %11 = and i16 %10, 1024
  %12 = shl i16 %a, 3
  %13 = and i16 %12, 512
  %14 = shl i16 %a, 1
  %15 = and i16 %14, 256
  %16 = lshr i16 %a, 1
  %17 = and i16 %16, 128
  %18 = lshr i16 %a, 3
  %19 = and i16 %18, 64
  %20 = lshr i16 %a, 5
  %21 = and i16 %20, 32
  %22 = lshr i16 %a, 7
  %23 = and i16 %22, 16
  %24 = lshr i16 %a, 9
  %25 = and i16 %24, 8
  %26 = lshr i16 %a, 11
  %27 = and i16 %26, 4
  %28 = lshr i16 %a, 13
  %29 = and i16 %28, 2
  %30 = lshr i16 %a, 15
  %31 = or i16 %30, %1
  %32 = or i16 %31, %3
  %33 = or i16 %32, %5
  %34 = or i16 %33, %7
  %35 = or i16 %34, %9
  %36 = or i16 %35, %11
  %37 = or i16 %36, %13
  %38 = or i16 %37, %15
  %39 = or i16 %38, %17
  %40 = or i16 %39, %19
  %41 = or i16 %40, %21
  %42 = or i16 %41, %23
  %43 = or i16 %42, %25
  %44 = or i16 %43, %27
  %45 = or i16 %44, %29
  ret i16 %45
}