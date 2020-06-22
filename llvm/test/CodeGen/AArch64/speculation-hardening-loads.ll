; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu | FileCheck %s

define i128 @ldp_single_csdb(i128* %p) speculative_load_hardening {
entry:
  %0 = load i128, i128* %p, align 16
  ret i128 %0
; CHECK-LABEL: ldp_single_csdb
; CHECK:      ldp   x8, x1, [x0]
; CHECK-NEXT: cmp sp, #0
; CHECK-NEXT: csetm x16, ne
; CHECK-NEXT: and   x8, x8, x16
; CHECK-NEXT: and   x1, x1, x16
; CHECK-NEXT: csdb
; CHECK-NEXT: mov [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT: and [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT: mov x0, x8
; CHECK-NEXT: mov sp, [[TMPREG]]
; CHECK-NEXT: ret
}

define double @ld_double(double* %p) speculative_load_hardening {
entry:
  %0 = load double, double* %p, align 8
  ret double %0
; Checking that the address laoded from is masked for a floating point load.
; CHECK-LABEL: ld_double
; CHECK:      cmp sp, #0
; CHECK-NEXT: csetm x16, ne
; CHECK-NEXT: and   x0, x0, x16
; CHECK-NEXT: csdb
; CHECK-NEXT: ldr   d0, [x0]
; CHECK-NEXT: mov [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT: and [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT: mov sp, [[TMPREG]]
; CHECK-NEXT: ret
}

define i32 @csdb_emitted_for_subreg_use(i64* %p, i32 %b) speculative_load_hardening {
entry:
  %X = load i64, i64* %p, align 8
  %X_trunc = trunc i64 %X to i32
  %add = add i32 %b, %X_trunc
  %iszero = icmp eq i64 %X, 0
  %ret = select i1 %iszero, i32 %b, i32 %add
  ret i32 %ret
; Checking that the address laoded from is masked for a floating point load.
; CHECK-LABEL: csdb_emitted_for_subreg_use
; CHECK:      ldr x8, [x0]
; CHECK-NEXT: cmp sp, #0
; CHECK-NEXT: csetm x16, ne
; CHECK-NEXT: and x8, x8, x16
; csdb instruction must occur before the add instruction with w8 as operand.
; CHECK-NEXT: csdb
; CHECK-NEXT: add w9, w1, w8
; CHECK-NEXT: cmp x8, #0
; CHECK-NEXT: csel w0, w1, w9, eq
; CHECK-NEXT: mov [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT: and [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT: mov sp, [[TMPREG]]
; CHECK-NEXT: ret
}

define i64 @csdb_emitted_for_superreg_use(i32* %p, i64 %b) speculative_load_hardening {
entry:
  %X = load i32, i32* %p, align 4
  %X_ext = zext i32 %X to i64
  %add = add i64 %b, %X_ext
  %iszero = icmp eq i32 %X, 0
  %ret = select i1 %iszero, i64 %b, i64 %add
  ret i64 %ret
; Checking that the address laoded from is masked for a floating point load.
; CHECK-LABEL: csdb_emitted_for_superreg_use
; CHECK:      ldr w8, [x0]
; CHECK-NEXT: cmp sp, #0
; CHECK-NEXT: csetm x16, ne
; CHECK-NEXT: and w8, w8, w16
; csdb instruction must occur before the add instruction with x8 as operand.
; CHECK-NEXT: csdb
; CHECK-NEXT: add x9, x1, x8
; CHECK-NEXT: cmp w8, #0
; CHECK-NEXT: csel x0, x1, x9, eq
; CHECK-NEXT: mov [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT: and [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT: mov sp, [[TMPREG]]
; CHECK-NEXT: ret
}

define i64 @no_masking_with_full_control_flow_barriers(i64 %a, i64 %b, i64* %p) speculative_load_hardening {
; CHECK-LABEL: no_masking_with_full_control_flow_barriers
; CHECK: dsb sy
; CHECK: isb
entry:
  %0 = tail call i64 asm "hint #12", "={x17},{x16},0"(i64 %b, i64 %a)
  %X = load i64, i64* %p, align 8
  %ret = add i64 %X, %0
; CHECK-NOT: csdb
; CHECK-NOT: and
; CHECK: ret
  ret i64 %ret
}

define void @f_implicitdef_vector_load(<4 x i32>* %dst, <2 x i32>* %src) speculative_load_hardening
{
entry:
  %0 = load <2 x i32>, <2 x i32>* %src, align 8
  %shuffle = shufflevector <2 x i32> %0, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  store <4 x i32> %shuffle, <4 x i32>* %dst, align 4
  ret void
; CHECK-LABEL: f_implicitdef_vector_load
; CHECK:       cmp     sp, #0
; CHECK-NEXT:  csetm   x16, ne
; CHECK-NEXT:  and     x1, x1, x16
; CHECK-NEXT:  csdb
; CHECK-NEXT:  ldr     d0, [x1]
; CHECK-NEXT:  mov     v0.d[1], v0.d[0]
; CHECK-NEXT:  str     q0, [x0]
; CHECK-NEXT:  mov     [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT:  and     [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT:  mov     sp, [[TMPREG]]
; CHECK-NEXT:  ret
}

define <2 x double> @f_usedefvectorload(double* %a, double* %b) speculative_load_hardening {
entry:
; CHECK-LABEL: f_usedefvectorload
; CHECK:       cmp     sp, #0
; CHECK-NEXT:  csetm   x16, ne
; CHECK-NEXT:  movi    v0.2d, #0000000000000000
; CHECK-NEXT:  and     x1, x1, x16
; CHECK-NEXT:  csdb
; CHECK-NEXT:  ld1     { v0.d }[0], [x1]
; CHECK-NEXT:  mov     [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT:  and     [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT:  mov     sp, [[TMPREG]]
; CHECK-NEXT:  ret
  %0 = load double, double* %b, align 16
  %vld1_lane = insertelement <2 x double> <double undef, double 0.000000e+00>, double %0, i32 0
  ret <2 x double> %vld1_lane
}

define i32 @deadload() speculative_load_hardening {
entry:
; CHECK-LABEL: deadload
; CHECK:       cmp     sp, #0
; CHECK-NEXT:  csetm   x16, ne
; CHECK-NEXT:  sub     sp, sp, #16
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  ldr     w8, [sp, #12]
; CHECK-NEXT:  add     sp, sp, #16
; CHECK-NEXT:  mov     [[TMPREG:x[0-9]+]], sp
; CHECK-NEXT:  and     [[TMPREG]], [[TMPREG]], x16
; CHECK-NEXT:  mov     sp, [[TMPREG]]
; CHECK-NEXT:  ret
  %a = alloca i32, align 4
  %val = load volatile i32, i32* %a, align 4
  ret i32 undef
}
