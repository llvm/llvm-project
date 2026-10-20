; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; 32-bit division by a variable: a reciprocal estimate, one Newton step with
; multiply-highs and a correction loop that a zero divisor skips; no loop
; over the quotient bits. 64-bit division checks at run time whether both
; operands fit in 32 bits.

; CHECK-LABEL: udiv32:
; CHECK:         rcp
; CHECK:         imadd {{r[0-9]+_r[0-9]+}}
; CHECK:         stop
define agx_kernel void @udiv32(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %q = udiv i32 %a, %b
  store i32 %q, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: srem32:
; CHECK:         rcp
; CHECK:         stop
define agx_kernel void @srem32(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %r = srem i32 %a, %b
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: udiv64:
; CHECK:         rcp
; CHECK:         stop
define agx_kernel void @udiv64(ptr addrspace(1) %p, i64 %a, i64 %b) {
  %q = udiv i64 %a, %b
  store i64 %q, ptr addrspace(1) %p
  ret void
}
