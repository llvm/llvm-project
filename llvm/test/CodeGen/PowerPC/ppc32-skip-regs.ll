; RUN: llc -O2 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-buildroot-linux-gnu"

@x = global ppc_fp128 0xM3FF00000000000000000000000000000, align 16
@.str = private unnamed_addr constant [9 x i8] c"%Lf %Lf\0A\00", align 1

define void @foo() #0 {
entry:
  %0 = load ppc_fp128, ppc_fp128* @x, align 16
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0), ppc_fp128 %0, ppc_fp128 %0)
  ret void
}
; Do not put second argument of function in r8 register, because there is no enough registers
; left for long double type (4 registers in soft float mode). Instead in r8 register this
; argument put on stack.
; CHECK-NOT: mr 8, 4
; CHECK: stw 6, 16(1)
; CHECK: stw 7, 20(1)
; CHECK: stw 5, 12(1)
; CHECK: stw 4, 8(1)

declare i32 @printf(i8* nocapture readonly, ...)

attributes #0 = { "use-soft-float"="true" }
