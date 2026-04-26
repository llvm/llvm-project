; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r2 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel 2>&1 | FileCheck %s

; CHECK:      FastISel missed call:
; CHECK-SAME: %call = call fastcc i32 @foo(i32 signext %a, i32 signext %b)

define internal i32 @bar(i32 signext %a, i32 signext %b) {
  %s = and i32 %a, %b
  ret i32 %s
}

define i32 @foo(i32 signext %a, i32 signext %b) {
  %call = call fastcc i32 @foo(i32 signext %a, i32 signext %b)
  ret i32 %call
}
