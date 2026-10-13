; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck -check-prefix=CHECK-DEFAULT %s
; RUN: llc -mtriple=hexagon -mattr=+reserved-r6 -O2 < %s | FileCheck -check-prefix=CHECK-R6 %s
; RUN: llc -mtriple=hexagon -mattr=+reserved-r6,+reserved-r7 -O2 < %s | FileCheck -check-prefix=CHECK-R6R7 %s
; RUN: llc -mtriple=hexagon -mattr=+reserved-r16 -O2 < %s | FileCheck -check-prefix=CHECK-R16 %s
; RUN: llc -mtriple=hexagon -mattr=+reserved-r16,+reserved-r17 -O2 < %s | FileCheck -check-prefix=CHECK-R16R17 %s

; Test that reserved registers are not used by the register allocator.
; The function has a call, forcing values to be placed in callee-saved
; registers (R16-R27). Reserving a register must prevent its use.
; Caller-saved registers (R6-R15) can also be reserved.

declare void @bar()

define i32 @pressure(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
  %v1 = add i32 %a, %b
  %v2 = mul i32 %v1, %c
  %v3 = sub i32 %v2, %d
  %v4 = add i32 %v3, %e
  %v5 = mul i32 %v4, %f
  call void @bar()
  %v6 = add i32 %v5, %v1
  %v7 = sub i32 %v6, %v2
  %v8 = add i32 %v7, %v3
  %v9 = mul i32 %v8, %v4
  %v10 = sub i32 %v9, %v5
  ret i32 %v10
}

; CHECK-DEFAULT: r16
; CHECK-DEFAULT: r17
; CHECK-R6-NOT: r6
; CHECK-R6R7-NOT: r6
; CHECK-R6R7-NOT: r7
; CHECK-R16-NOT: r16
; CHECK-R16R17-NOT: r16
; CHECK-R16R17-NOT: r17
