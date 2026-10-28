; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=RCP
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-launch-divisors < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-launch-divisors -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s --check-prefix=SECTION

; Division by a 32-bit kernel argument d with -agx-launch-divisors: the
; launch computes d's multiplier m and shifts s1, s2 into the uniforms
; (.agx.kernels extension 7) and the kernel computes
; q = (t + ((x - t) >> s1)) >> s2, t = mulhi(x, m), instead of a reciprocal
; and two corrections.

; RCP-LABEL: udiv_arg:
; RCP:         rcp
; CHECK-LABEL: udiv_arg:
; CHECK-NOT:   rcp
; CHECK:       imadd [[T:r[0-9]+]]_[[HI:r[0-9]+]], u2, u7, 0
; CHECK-NEXT:  isub [[D:r[0-9]+]], u2, [[HI]]
; CHECK-NEXT:  bfeil [[D]], 0, [[D]], u8
; CHECK-NEXT:  iadd [[S:r[0-9]+]], [[HI]], [[D]]
; CHECK:       bfeil [[Q:r[0-9]+]], 0, [[S]], u9
; CHECK:       icmpsel ueq, {{r[0-9]+}}, u3, 0,
; Extension 7, 3 values of argument 2 (half 6): m, s1, s2 at halves 14, 16, 18.
; SECTION: 07000000 03000000 0e000000
; SECTION: 06000000 00000000 10000000 06000000
; SECTION: 01000000 12000000 06000000 02000000
define agx_kernel void @udiv_arg(ptr addrspace(1) %p, i32 %x, i32 %d) {
  %q = udiv i32 %x, %d
  %r = urem i32 %x, %d
  %s = add i32 %q, %r
  store i32 %s, ptr addrspace(1) %p
  ret void
}

; A signed divisor: the values of |d| (kinds 3 to 5).
; CHECK-LABEL: sdiv_arg:
; CHECK-NOT:   rcp
; CHECK:       u7
; SECTION: 07000000 03000000 0e000000
; SECTION: 06000000 03000000 10000000 06000000
; SECTION: 04000000 12000000 06000000 05000000
define agx_kernel void @sdiv_arg(ptr addrspace(1) %p, i32 %x, i32 %d) {
  %q = sdiv i32 %x, %d
  store i32 %q, ptr addrspace(1) %p
  ret void
}

; The high word of a 64-bit argument (half 6 + 2), as the 32-bit fast path of
; a 64-bit division takes it.
; CHECK-LABEL: udiv_high_word:
; CHECK-NOT:   rcp
; CHECK:       icmpsel ueq, {{r[0-9]+}}, u4, 0,
define agx_kernel void @udiv_high_word(ptr addrspace(1) %p, i32 %x, i64 %d) {
  %f = freeze i64 %d
  %h = lshr i64 %f, 32
  %t = trunc i64 %h to i32
  %q = udiv i32 %x, %t
  store i32 %q, ptr addrspace(1) %p
  ret void
}

; Another divisor keeps the reciprocal.
; CHECK-LABEL: udiv_not_arg:
; CHECK:       rcp
define agx_kernel void @udiv_not_arg(ptr addrspace(1) %p, i32 %x, i32 %d) {
  %e = add i32 %d, 1
  %q = udiv i32 %x, %e
  store i32 %q, ptr addrspace(1) %p
  ret void
}
