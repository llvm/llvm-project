; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; 32-bit division by a variable: a reciprocal estimate, one Newton step with
; multiply-highs and two corrections, without control flow; no loop over the
; quotient bits. 64-bit division checks at run time whether both operands fit
; in 32 bits.

; CHECK-LABEL: udiv32:
; CHECK-NOT:     while_
; CHECK:         rcp
; CHECK:         imadd {{r[0-9]+_r[0-9]+}}
; CHECK-NOT:     while_
; CHECK-NOT:     jmp_exec_any
; CHECK:         stop
define agx_kernel void @udiv32(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %q = udiv i32 %a, %b
  store i32 %q, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: srem32:
; CHECK:         rcp
; CHECK-NOT:     jmp_exec
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

; Divisions by the same divisor share its reciprocal, also when each has its
; own copy of the divisor (umax(y, 1) in each unrolled iteration, behind each
; 64-bit division's 32-bit fast path).
; CHECK-LABEL: shared_divisor:
; CHECK:         rcp
; CHECK-NOT:     rcp
; CHECK:         stop
define agx_kernel void @shared_divisor(ptr addrspace(1) %p, i32 %a, i32 %b, i32 %y, i1 %c) {
entry:
  %y1 = call i32 @llvm.umax.i32(i32 %y, i32 1)
  %q1 = udiv i32 %a, %y1
  store i32 %q1, ptr addrspace(1) %p
  br i1 %c, label %then, label %end
then:
  %y2 = call i32 @llvm.umax.i32(i32 %y, i32 1)
  %r2 = urem i32 %b, %y2
  %p2 = getelementptr i32, ptr addrspace(1) %p, i64 1
  store i32 %r2, ptr addrspace(1) %p2
  br label %end
end:
  ret void
}
