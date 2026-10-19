; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; The hardware keeps 16-bit denormals. Functions whose denormal mode for half
; flushes get flush(x) = |x| < 2^-14 ? copysign(0, x) : x on the operands and
; the results of half arithmetic.

; CHECK-LABEL: flush:
; CHECK:         mov_imm [[SIGN:r[0-9]+[lh]]], 32768
; CHECK:         mov_imm [[MIN:r[0-9]+[lh]]], 1024
; CHECK:         fcmpsel lt, {{r[0-9]+[lh]}}, u2l.abs, [[MIN]],
; CHECK:         fcmpsel lt, {{r[0-9]+[lh]}}, u2h.abs, [[MIN]],
; CHECK:         fadd16 [[SUM:r[0-9]+[lh]]],
; CHECK:         and {{r[0-9]+[lh]}}, [[SUM]], [[SIGN]]
; CHECK:         fcmpsel lt, {{r[0-9]+[lh]}}, [[SUM]].abs, [[MIN]],
; CHECK:         device_store
define agx_kernel void @flush(ptr addrspace(1) %out, half %a, half %b) #0 {
  %s = fadd half %a, %b
  store half %s, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: ieee:
; CHECK-NOT:     fcmpsel
; CHECK:         fadd16
; CHECK-NEXT:    device_store
define agx_kernel void @ieee(ptr addrspace(1) %out, half %a, half %b) {
  %s = fadd half %a, %b
  store half %s, ptr addrspace(1) %out
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) }
