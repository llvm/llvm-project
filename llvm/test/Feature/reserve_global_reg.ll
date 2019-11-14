; RUN: not llc < %s -mtriple=thumbv7-apple-darwin -mattr=+reserve-r7 -o - 2>&1 | FileCheck -check-prefix=CHECK-RESERVE-FP7 %s
; RUN: not llc < %s -mtriple=armv7-windows-msvc -mattr=+reserve-r11 -o - 2>&1 | FileCheck -check-prefix=CHECK-RESERVE-FP11 %s
; RUN: not llc < %s -mtriple=thumbv7-windows -mattr=+reserve-r11 -o - 2>&1 | FileCheck -check-prefix=CHECK-RESERVE-FP11-2 %s

; int test(int a, int b, int c) {
;   return a + b + c;
; }

; Function Attrs: noinline nounwind optnone
define hidden i32 @_Z4testiii(i32 %a, i32 %b, i32 %c) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  store i32 %c, i32* %c.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* %c.addr, align 4
  %add1 = add nsw i32 %add, %2
  ret i32 %add1
}

; CHECK-RESERVE-FP7: Register r7 has been specified but is used as the frame pointer for this target.
; CHECK-RESERVE-FP11: Register r11 has been specified but is used as the frame pointer for this target.
; CHECK-RESERVE-FP11-2: Register r11 has been specified but is used as the frame pointer for this target.

