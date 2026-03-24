; Test atomic long double minimum.
; Expect a libcall in a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %ret, ptr %src, ptr %b) {
; CHECK-LABEL: f1:
; CHECK:       .cfi_startproc
; CHECK-NEXT:    # %bb.0:
; CHECK-NEXT:    stmg	%r12, %r15, 96(%r15)
; CHECK-NEXT:    .cfi_offset %r12, -64
; CHECK-NEXT:    .cfi_offset %r13, -56
; CHECK-NEXT:    .cfi_offset %r14, -48
; CHECK-NEXT:    .cfi_offset %r15, -40
; CHECK-NEXT:    aghi	%r15, -256
; CHECK-NEXT:    .cfi_def_cfa_offset 416
; CHECK-NEXT:    std	%f8, 248(%r15)                  # 8-byte Spill
; CHECK-NEXT:    std	%f9, 240(%r15)                  # 8-byte Spill
; CHECK-NEXT:    std	%f10, 232(%r15)                 # 8-byte Spill
; CHECK-NEXT:    std	%f11, 224(%r15)                 # 8-byte Spill
; CHECK-NEXT:    .cfi_offset %f8, -168
; CHECK-NEXT:    .cfi_offset %f9, -176
; CHECK-NEXT:    .cfi_offset %f10, -184
; CHECK-NEXT:    .cfi_offset %f11, -192
; CHECK-NEXT:    lgr	%r13, %r3
; CHECK-NEXT:    ld	%f8, 0(%r4)
; CHECK-NEXT:    ld	%f10, 8(%r4)
; CHECK-NEXT:    ld	%f9, 0(%r3)
; CHECK-NEXT:    ld	%f11, 8(%r3)
; CHECK-NEXT:    lgr	%r12, %r2
; CHECK-NEXT:    .LBB0_1:                                # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    std	%f8, 160(%r15)
; CHECK-NEXT:    std	%f10, 168(%r15)
; CHECK-NEXT:    std	%f9, 176(%r15)
; CHECK-NEXT:    std	%f11, 184(%r15)
; CHECK-NEXT:    la	%r2, 192(%r15)
; CHECK-NEXT:    la	%r3, 176(%r15)
; CHECK-NEXT:    la	%r4, 160(%r15)
; CHECK-NEXT:    brasl	%r14, fminl@PLT
; CHECK-NEXT:    lg	%r1, 200(%r15)
; CHECK-NEXT:    lg	%r0, 192(%r15)
; CHECK-NEXT:    lgdr	%r3, %f11
; CHECK-NEXT:    lgdr	%r2, %f9
; CHECK-NEXT:    cdsg	%r2, %r0, 0(%r13)
; CHECK-NEXT:    stg	%r3, 216(%r15)
; CHECK-NEXT:    stg	%r2, 208(%r15)
; CHECK-NEXT:    ld	%f9, 208(%r15)
; CHECK-NEXT:    ld	%f11, 216(%r15)
; CHECK-NEXT:    jl	.LBB0_1
; CHECK-NEXT:    # %bb.2:                                # %atomicrmw.end
; CHECK-NEXT:    std	%f9, 0(%r12)
; CHECK-NEXT:    std	%f11, 8(%r12)
; CHECK-NEXT:    ld	%f8, 248(%r15)                  # 8-byte Reload
; CHECK-NEXT:    ld	%f9, 240(%r15)                  # 8-byte Reload
; CHECK-NEXT:    ld	%f10, 232(%r15)                 # 8-byte Reload
; CHECK-NEXT:    ld	%f11, 224(%r15)                 # 8-byte Reload
; CHECK-NEXT:    lmg	%r12, %r15, 352(%r15)
; CHECK-NEXT:    br	%r14
  %val = load fp128, ptr %b
  %res = atomicrmw fmin ptr %src, fp128 %val seq_cst
  store fp128 %res, ptr %ret
  ret void
}
