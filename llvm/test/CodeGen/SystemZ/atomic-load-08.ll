; Test long double atomic loads. These are emitted by the Clang FE as i128
; loads with a bitcast, and this test case gets converted into that form as
; well by the AtomicExpand pass.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck -check-prefixes=CHECK,BASE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck -check-prefixes=CHECK,Z13 %s

define void @f1(ptr %ret, ptr %src) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lpq %r0, 0(%r3)
; CHECK-NEXT:    stg %r1, 8(%r2)
; CHECK-NEXT:    stg %r0, 0(%r2)
; CHECK-NEXT:    br %r14
  %val = load atomic fp128, ptr %src seq_cst, align 16
  store fp128 %val, ptr %ret, align 8
  ret void
}

define void @f1_fpuse(ptr %ret, ptr %src) {
; CHECK-LABEL: f1_fpuse:
; CHECK:       # %bb.0:
; BASE-NEXT: aghi	%r15, -176
; BASE-NEXT: .cfi_def_cfa_offset 336

; CHECK-NEXT:	lpq	%r0, 0(%r3)

; BASE-NEXT: stg %r1, 168(%r15)
; BASE-NEXT: stg %r0, 160(%r15)
; BASE-NEXT: ld	%f0, 160(%r15)
; BASE-NEXT: ld	%f2, 168(%r15)

; Z13-NEXT: vlvgp %v0, %r0, %r1
; Z13-NEXT: vrepg %v2, %v0, 1

; CHECK-NEXT:	axbr	%f0, %f0
; CHECK-NEXT:	std	%f0, 0(%r2)
; CHECK-NEXT:	std	%f2, 8(%r2)
; BASE-NEXT:	aghi	%r15, 176
; CHECK-NEXT:	br	%r14

  %val = load atomic fp128, ptr %src seq_cst, align 16
  %use = fadd fp128 %val, %val
  store fp128 %use, ptr %ret, align 8
  ret void
}

define void @f2(ptr %ret, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __atomic_load@PLT
  %val = load atomic fp128, ptr %src seq_cst, align 8
  store fp128 %val, ptr %ret, align 8
  ret void
}

define void @f2_fpuse(ptr %ret, ptr %src) {
; CHECK-LABEL: f2_fpuse:
; CHECK: brasl %r14, __atomic_load@PLT
; CHECK-NEXT:   ld	%f0, 160(%r15)
; CHECK-NEXT:	ld	%f2, 168(%r15)
; CHECK-NEXT:	axbr	%f0, %f0
; CHECK-NEXT:	std	%f0, 0(%r13)
; CHECK-NEXT:	std	%f2, 8(%r13)
; CHECK-NEXT:	lmg	%r13, %r15, 280(%r15)
; CHECK-NEXT:	br	%r14
  %val = load atomic fp128, ptr %src seq_cst, align 8
  %use = fadd fp128 %val, %val
  store fp128 %use, ptr %ret, align 8
  ret void
}
