; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -verify-machineinstrs < %s | FileCheck %s

; Any function of two sources is one bitop. xnor, nand and nor print by name,
; the others print their table from bit 0 (bit b * 2 + a: the result for
; source bits a and b).

; CHECK-LABEL: andn:
; CHECK:         bitop 0100, r{{[0-9]+}}, u2, u3
define agx_kernel void @andn(ptr addrspace(1) %o, i32 %a, i32 %b) {
  %n = xor i32 %b, -1
  %r = and i32 %a, %n
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: orn:
; CHECK:         bitop 1101, r{{[0-9]+}}, u2, u3
define agx_kernel void @orn(ptr addrspace(1) %o, i32 %a, i32 %b) {
  %n = xor i32 %b, -1
  %r = or i32 %a, %n
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: xnor:
; CHECK:         xnor r{{[0-9]+}}, u2, u3
define agx_kernel void @xnor(ptr addrspace(1) %o, i32 %a, i32 %b) {
  %x = xor i32 %a, %b
  %r = xor i32 %x, -1
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: nand:
; CHECK:         nand r{{[0-9]+}}, u2, u3
define agx_kernel void @nand(ptr addrspace(1) %o, i32 %a, i32 %b) {
  %x = and i32 %a, %b
  %r = xor i32 %x, -1
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: nor:
; CHECK:         nor r{{[0-9]+}}, u2, u3
define agx_kernel void @nor(ptr addrspace(1) %o, i32 %a, i32 %b) {
  %x = or i32 %a, %b
  %r = xor i32 %x, -1
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; A low mask wider than an 8-bit immediate is a bfeil, without a mov_imm.
; CHECK-LABEL: low_mask:
; CHECK-NOT:     mov_imm
; CHECK:         bfeil r{{[0-9]+}}, 0, u2, 0, mask 0xFFFFF
define agx_kernel void @low_mask(ptr addrspace(1) %o, i32 %a) {
  %r = and i32 %a, 1048575
  store i32 %r, ptr addrspace(1) %o
  ret void
}

; A mask with other uses stays one mov_imm and an and per use.
; CHECK-LABEL: shared_mask:
; CHECK:         mov_imm [[M:r[0-9]+]], 1048575
; CHECK:         and {{r[0-9]+}}, u2, [[M]]
; CHECK:         and {{r[0-9]+}}, u3, [[M]]
define agx_kernel void @shared_mask(ptr addrspace(1) %o, i32 %a, i32 %b) {
  %x = and i32 %a, 1048575
  %y = and i32 %b, 1048575
  %r = mul i32 %x, %y
  store i32 %r, ptr addrspace(1) %o
  ret void
}
