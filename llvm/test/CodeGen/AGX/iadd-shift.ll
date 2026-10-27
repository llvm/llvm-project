; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -verify-machineinstrs < %s | FileCheck %s

; iadd / isub / imadd shift their second source (imadd: the addend) left by
; 0 to 4 bits.

; CHECK-LABEL: add_shl:
; CHECK:         iadd r{{[0-9]+}}, u3, u2, lsl 3
define agx_kernel void @add_shl(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %s = shl i32 %a, 3
  %r = add i32 %b, %s
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: sub_shl:
; CHECK:         isub r{{[0-9]+}}, u3, u2, lsl 4
define agx_kernel void @sub_shl(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %s = shl i32 %a, 4
  %r = sub i32 %b, %s
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; A shift by more than 4 is a separate instruction.
; CHECK-LABEL: add_shl5:
; CHECK:         bfi
; CHECK-NEXT:    iadd r{{[0-9]+}}, r{{[0-9]+}}, u3{{$}}
define agx_kernel void @add_shl5(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %s = shl i32 %a, 5
  %r = add i32 %s, %b
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: mad_shl:
; CHECK:         imadd r{{[0-9]+}}, u2, u3, u4, lsl 2
define agx_kernel void @mad_shl(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c) {
  %m = mul i32 %a, %b
  %s = shl i32 %c, 2
  %r = add i32 %m, %s
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; x * (2^k + 1), k <= 4: x + (x << k).
; CHECK-LABEL: mul9:
; CHECK:         iadd r{{[0-9]+}}, u2, u2, lsl 3
; CHECK-LABEL: mul7:
; CHECK:         imadd r{{[0-9]+}}, u2, 7, 0
define agx_kernel void @mul9(ptr addrspace(1) %out, i32 %a) {
  %r = mul i32 %a, 9
  store i32 %r, ptr addrspace(1) %out
  ret void
}
define agx_kernel void @mul7(ptr addrspace(1) %out, i32 %a) {
  %r = mul i32 %a, 7
  store i32 %r, ptr addrspace(1) %out
  ret void
}
