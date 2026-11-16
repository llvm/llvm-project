; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare ppc_fp128 @llvm.ldexp.ppcf128.i32(ppc_fp128, i32)

; Positive overflow.
; CHECK-LABEL: define ppc_fp128 @positive_overflow()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret ppc_fp128 +inf
define ppc_fp128 @positive_overflow() {
entry:
  %r = call ppc_fp128 @llvm.ldexp.ppcf128.i32(
    ppc_fp128 f0xC0900000000000007E70000000000000,
    i32 100)
  ret ppc_fp128 %r
}

; Negative overflow.
; CHECK-LABEL: define ppc_fp128 @negative_overflow()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret ppc_fp128 -inf
define ppc_fp128 @negative_overflow() {
entry:
  %r = call ppc_fp128 @llvm.ldexp.ppcf128.i32(
    ppc_fp128 f0x4090000000000000FE70000000000000,
    i32 100)
  ret ppc_fp128 %r
}

; Finite value.
; CHECK-LABEL: define ppc_fp128 @finite_value()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret ppc_fp128 1.000000e+00
define ppc_fp128 @finite_value() {
entry:
  %r = call ppc_fp128 @llvm.ldexp.ppcf128.i32(
    ppc_fp128 f0x00000000000000003FF0000000000000,
    i32 0)
  ret ppc_fp128 %r
}
