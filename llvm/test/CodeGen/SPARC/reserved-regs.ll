; RUN: llc -march=sparc -verify-machineinstrs < %s | FileCheck %s

@g = common global [32 x i32] zeroinitializer, align 16
@h = common global [16 x i64] zeroinitializer, align 16

;; Ensures that we don't use registers which are supposed to be reserved.

; CHECK-LABEL: use_all_i32_regs:
; CHECK: save %sp
; CHECK: .cfi_register %o7, %i7
; CHECK-NOT: %g0
; CHECK-NOT: %g1
; CHECK-NOT: %g5
; CHECK-NOT: %g6
; CHECK-NOT: %g7
; CHECK-NOT: %o6
; CHECK-NOT: %i6
; CHECK-NOT: %i7
; CHECK: ret
define void @use_all_i32_regs() {
entry:
  %0 = load volatile i32, ptr @g, align 16
  %1 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 1), align 4
  %2 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 2), align 8
  %3 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 3), align 4
  %4 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 4), align 16
  %5 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 5), align 4
  %6 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 6), align 8
  %7 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 7), align 4
  %8 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 8), align 16
  %9 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 9), align 4
  %10 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 10), align 8
  %11 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 11), align 4
  %12 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 12), align 16
  %13 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 13), align 4
  %14 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 14), align 8
  %15 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 15), align 4
  %16 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 16), align 16
  %17 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 17), align 4
  %18 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 18), align 8
  %19 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 19), align 4
  %20 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 20), align 16
  %21 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 21), align 4
  %22 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 22), align 8
  %23 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 23), align 4
  %24 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 24), align 16
  %25 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 25), align 4
  %26 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 26), align 8
  %27 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 27), align 4
  %28 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 28), align 16
  %29 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 29), align 4
  %30 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 30), align 8
  %31 = load volatile i32, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 31), align 4
  store volatile i32 %1, ptr @g, align 16
  store volatile i32 %2, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 1), align 4
  store volatile i32 %3, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 2), align 8
  store volatile i32 %4, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 3), align 4
  store volatile i32 %5, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 4), align 16
  store volatile i32 %6, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 5), align 4
  store volatile i32 %7, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 6), align 8
  store volatile i32 %8, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 7), align 4
  store volatile i32 %9, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 8), align 16
  store volatile i32 %10, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 9), align 4
  store volatile i32 %11, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 10), align 8
  store volatile i32 %12, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 11), align 4
  store volatile i32 %13, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 12), align 16
  store volatile i32 %14, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 13), align 4
  store volatile i32 %15, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 14), align 8
  store volatile i32 %16, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 15), align 4
  store volatile i32 %17, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 16), align 16
  store volatile i32 %18, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 17), align 4
  store volatile i32 %19, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 18), align 8
  store volatile i32 %20, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 19), align 4
  store volatile i32 %21, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 20), align 16
  store volatile i32 %22, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 21), align 4
  store volatile i32 %23, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 22), align 8
  store volatile i32 %24, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 23), align 4
  store volatile i32 %25, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 24), align 16
  store volatile i32 %26, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 25), align 4
  store volatile i32 %27, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 26), align 8
  store volatile i32 %28, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 27), align 4
  store volatile i32 %29, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 28), align 16
  store volatile i32 %30, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 29), align 4
  store volatile i32 %31, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 30), align 8
  store volatile i32 %0, ptr getelementptr inbounds ([32 x i32], ptr @g, i64 0, i64 31), align 4
  ret void
}


; CHECK-LABEL: use_all_i64_regs:
; CHECK: save %sp
; CHECK: .cfi_register %o7, %i7
; CHECK-NOT: %g0
; CHECK-NOT: %g1
; CHECK-NOT: %g4
; CHECK-NOT: %g5
; CHECK-NOT: %g6
; CHECK-NOT: %g7
; CHECK-NOT: %o6
; CHECK-NOT: %o7
; CHECK-NOT: %i6
; CHECK-NOT: %i7
; CHECK: ret
define void @use_all_i64_regs() {
entry:
  %0 = load volatile i64, ptr @h, align 16
  %1 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 1), align 4
  %2 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 2), align 8
  %3 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 3), align 4
  %4 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 4), align 16
  %5 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 5), align 4
  %6 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 6), align 8
  %7 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 7), align 4
  %8 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 8), align 16
  %9 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 9), align 4
  %10 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 10), align 8
  %11 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 11), align 4
  %12 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 12), align 16
  %13 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 13), align 4
  %14 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 14), align 8
  %15 = load volatile i64, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 15), align 4
  store volatile i64 %1, ptr @h, align 16
  store volatile i64 %2, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 1), align 4
  store volatile i64 %3, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 2), align 8
  store volatile i64 %4, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 3), align 4
  store volatile i64 %5, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 4), align 16
  store volatile i64 %6, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 5), align 4
  store volatile i64 %7, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 6), align 8
  store volatile i64 %8, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 7), align 4
  store volatile i64 %9, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 8), align 16
  store volatile i64 %10, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 9), align 4
  store volatile i64 %11, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 10), align 8
  store volatile i64 %12, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 11), align 4
  store volatile i64 %13, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 12), align 16
  store volatile i64 %14, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 13), align 4
  store volatile i64 %15, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 14), align 8
  store volatile i64 %0, ptr getelementptr inbounds ([16 x i64], ptr @h, i64 0, i64 15), align 4
  ret void
}
