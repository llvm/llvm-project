; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600 %s
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI %s

; BFI_INT Definition pattern from ISA docs
; (y & x) | (z & ~x)
;
; R600: {{^}}bfi_def:
; R600: BFI_INT
; SI:   @bfi_def
; SI:   v_bfi_b32
define void @bfi_def(i32 addrspace(1)* %out, i32 %x, i32 %y, i32 %z) {
entry:
  %0 = xor i32 %x, -1
  %1 = and i32 %z, %0
  %2 = and i32 %y, %x
  %3 = or i32 %1, %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; SHA-256 Ch function
; z ^ (x & (y ^ z))
; R600: {{^}}bfi_sha256_ch:
; R600: BFI_INT
; SI:   @bfi_sha256_ch
; SI:   v_bfi_b32
define void @bfi_sha256_ch(i32 addrspace(1)* %out, i32 %x, i32 %y, i32 %z) {
entry:
  %0 = xor i32 %y, %z
  %1 = and i32 %x, %0
  %2 = xor i32 %z, %1
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; SHA-256 Ma function
; ((x & z) | (y & (x | z)))
; R600: {{^}}bfi_sha256_ma:
; R600: XOR_INT * [[DST:T[0-9]+\.[XYZW]]], KC0[2].Z, KC0[2].W
; R600: BFI_INT * {{T[0-9]+\.[XYZW]}}, {{[[DST]]|PV\.[XYZW]}}, KC0[3].X, KC0[2].W
; SI: v_xor_b32_e32 [[DST:v[0-9]+]], {{s[0-9]+, v[0-9]+}}
; SI: v_bfi_b32 {{v[0-9]+}}, [[DST]], {{s[0-9]+, v[0-9]+}}

define void @bfi_sha256_ma(i32 addrspace(1)* %out, i32 %x, i32 %y, i32 %z) {
entry:
  %0 = and i32 %x, %z
  %1 = or i32 %x, %z
  %2 = and i32 %y, %1
  %3 = or i32 %0, %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}
