; RUN: llc -mtriple i686-windows-itanium -filetype asm -o - %s | FileCheck %s

define i64 @f(i64 %i, i64 %j) {
  %1 = sdiv i64 %i, %j
  ret i64 %1
}

; CHECK-LABEL: _f:
; CHECK-NOT: calll ___divdi3
; CHECK: calll __alldiv

define i64 @g(i64 %i, i64 %j) {
  %1 = udiv i64 %i, %j
  ret i64 %1
}

; CHECK-LABEL: _g:
; CHECK-NOT: calll ___udivdi3
; CHECK: calll __aulldiv

define i64 @h(i64 %i, i64 %j) {
  %1 = srem i64 %i, %j
  ret i64 %1
}

; CHECK-LABEL: _h:
; CHECK-NOT: calll ___moddi3
; CHECK: calll __allrem

define i64 @i(i64 %i, i64 %j) {
  %1 = urem i64 %i, %j
  ret i64 %1
}

; CHECK-LABEL: _i:
; CHECK-NOT: calll ___umoddi3
; CHECK: calll __aullrem

