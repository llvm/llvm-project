; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1

; Function Attrs: nounwind
define i64 @main() #0 {
entry:
  %x = alloca i64, align 8
  store i64 0, i64* %x, align 8
  %0 = call i64 asm sideeffect "ld       $0,$1\0A\09add${2:I}   $0,$0,$2", "=&r,*m,Ir"(i64* %x, i64 -1) #0
  ret i64 %0
}

; CHECK: ld
; CHECK-NOT: addi   3, 3, 4294967295
; CHECK: addi   3, 3, -1
; CHECK: blr

; Function Attrs: nounwind
declare signext i32 @printf(i8* nocapture readonly, ...) #0

attributes #0 = { nounwind }

