; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; Small constants are immediate sources: a negative addend becomes a
; subtraction, and min/max compare against the immediate directly.

; CHECK-LABEL: add_neg:
; CHECK-NOT:     mov_imm r{{[0-9]+}}, -
; CHECK:         isub r{{[0-9]+}}, u{{[0-9]+}}, 3
define agx_kernel void @add_neg(ptr addrspace(1) %o, i32 %x) {
  %r = add i32 %x, -3
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: add_neg64:
; CHECK:         isub r{{[0-9]+}}_r{{[0-9]+}}, {{[ru][0-9]+_[ru][0-9]+}}, 1
define agx_kernel void @add_neg64(ptr addrspace(1) %o, i64 %x) {
  %r = add i64 %x, -1
  store i64 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: clamp:
; CHECK-NOT:     mov_imm r{{[0-9]+}}, 0
; CHECK:         icmpsel sgt, r{{[0-9]+}}, u{{[0-9]+}}, 0, u{{[0-9]+}}, 0
; CHECK:         icmpsel {{u?|s}}lt, r{{[0-9]+}}, r{{[0-9]+}}, 255, r{{[0-9]+}}, 255
define agx_kernel void @clamp(ptr addrspace(1) %o, i32 %x) {
  %a = call i32 @llvm.smax.i32(i32 %x, i32 0)
  %b = call i32 @llvm.umin.i32(i32 %a, i32 255)
  store i32 %b, ptr addrspace(1) %o
  ret void
}

; Beyond the 8-bit immediate the constant is materialized.
; CHECK-LABEL: add_big_neg:
; CHECK:         mov_imm r{{[0-9]+}}, -300
define agx_kernel void @add_big_neg(ptr addrspace(1) %o, i32 %x) {
  %r = add i32 %x, -300
  store i32 %r, ptr addrspace(1) %o
  ret void
}

declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
