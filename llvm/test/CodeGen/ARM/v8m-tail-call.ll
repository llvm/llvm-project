; RUN: llc %s -o - -mtriple=thumbv8m.base | FileCheck %s

declare i32 @g(...)

declare i32 @h0(i32, i32, i32, i32)
define hidden i32 @f0() {
  %1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %2 = tail call i32 @h0(i32 %1, i32 1, i32 2, i32 3)
  ret i32 %2
; CHECK-LABEL: f0
; CHECK:      ldr     [[POP:r[4567]]], [sp
; CHECK-NEXT: mov     lr, [[POP]]
; CHECK-NEXT: pop     {{.*}}[[POP]]
; CHECK-NEXT: add     sp, #4
; CHECK-NEXT: b       h0
}

declare i32 @h1(i32)
define hidden i32 @f1() {
  %1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %2 = tail call i32 @h1(i32 %1)
  ret i32 %2
; CHECK-LABEL: f1
; CHECK: pop     {r7}
; CHECK: pop     {r1}
; CHECK: mov     lr, r1
; CHECK: b       h1
}

declare i32 @h2(i32, i32, i32, i32, i32)
define hidden i32 @f2(i32, i32, i32, i32, i32) {
  %6 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %10, label %8

  %9 = tail call i32 @h2(i32 %6, i32 %1, i32 %2, i32 %3, i32 %4)
  br label %10

  %11 = phi i32 [ %9, %8 ], [ -1, %5 ]
  ret i32 %11
; CHECK-LABEL: f2
; CHECK:      ldr     [[POP:r[4567]]], [sp
; CHECK-NEXT: mov     lr, [[POP]]
; CHECK-NEXT: pop     {{.*}}[[POP]]
; CHECK-NEXT: add     sp, #4
; CHECK-NEXT: b       h2
}
