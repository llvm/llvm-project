; RUN: llc -verify-machineinstrs -O2 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-buildroot-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"%Lf\0A\00", align 1

define i32 @main() #0 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, ppc_fp128 0xM3FF00000000000000000000000000000)
  ret i32 0
}

; First available register for long double argument is r4, so put
; Hi part in r4/r5, Lo part in r6/r7 (do not switch Hi/Lo parts)
; CHECK: lis 4, 16368
; CHECK-NOT: lis 6, 16368
; CHECK: li 5, 0
; CHECK: li 7, 0

declare i32 @printf(ptr nocapture readonly, ...)

attributes #0 = { "use-soft-float"="true" }

