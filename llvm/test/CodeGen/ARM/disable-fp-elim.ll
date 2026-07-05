; RUN: llc < %s -mtriple armv7-none-linux-gnueabi -fast-isel -O1 | FileCheck %s --check-prefix=DISABLE-FP-ELIM
; RUN: llc < %s -mtriple armv7-none-linux-gnueabi -frame-pointer=all -O1 | FileCheck %s --check-prefix=DISABLE-FP-ELIM
; RUN: llc < %s -mtriple armv7-none-linux-gnueabi -frame-pointer=none -O1 | FileCheck %s --check-prefix=ENABLE-FP-ELIM
; RUN: llc < %s -mtriple armv7-none-linux-gnueabi -frame-pointer=none -O0 | FileCheck %s --check-prefix=DISABLE-FP-ELIM

; Check that command line option "-frame-pointer=all" sets function
; attribute "frame-pointer"="all". Also, check frame pointer
; elimination is disabled when fast-isel is used.

; ENABLE-FP-ELIM-NOT: .setfp
; DISABLE-FP-ELIM: .setfp r11, sp

define i32 @foo1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %call = tail call i32 @foo2(i32 %a)
  %add = add i32 %c, %b
  %add1 = add i32 %add, %d
  %add2 = add i32 %add1, %e
  %add3 = add i32 %add2, %call
  ret i32 %add3
}

declare i32 @foo2(i32)
