; RUN: llc -mtriple thumbv4t-eabi    < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V4T
; RUN: llc -mtriple armv8m.base-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8M

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; Function Attrs: nounwind
define <4 x i32> @f() local_unnamed_addr #0 {
entry:
  %call = tail call i32 @h(i32 1)
  %call1 = tail call <4 x i32> @g(i32 %call, i32 2, i32 3, i32 4)
  ret <4 x i32> %call1
; CHECK: ldr r7, [sp, #4]
; CHECK-NEXT: mov lr, r7
; CHECK-NEXT: pop {r7}
; CHECK-NEXT: add sp, #4
; CHECK-V47: bx lr
; CHECK-V8M: b g
}

declare <4 x i32> @g(i32, i32, i32, i32) local_unnamed_addr

declare i32 @h(i32) local_unnamed_addr

attributes #0 = { "disable-tail-calls"="false" "frame-pointer"="all" }
