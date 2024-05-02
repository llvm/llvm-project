; Test long double atomic stores. The atomic store is converted to i128 by
; the AtomicExpand pass.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck -check-prefixes=CHECK,BASE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck -check-prefixes=CHECK,Z13 %s

define void @f1(ptr %dst, ptr %src) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lg %r1, 8(%r3)
; CHECK-NEXT:    lg %r0, 0(%r3)
; CHECK-NEXT:    stpq %r0, 0(%r2)
; CHECK-NEXT:    bcr 1{{[45]}}, %r0
; CHECK-NEXT:    br %r14
  %val = load fp128, ptr %src, align 8
  store atomic fp128 %val, ptr %dst seq_cst, align 16
  ret void
}

define void @f1_fpsrc(ptr %dst, ptr %src) {
; CHECK-LABEL: f1_fpsrc:
; CHECK:       # %bb.0:
; CHECK-NEXT: ld	%f0, 0(%r3)
; CHECK-NEXT: ld	%f2, 8(%r3)
; CHECK-NEXT: axbr	%f0, %f0

; BASE-NEXT: lgdr	%r1, %f2
; BASE-NEXT: lgdr	%r0, %f0

; Z13-NEXT: vmrhg	%v0, %v0, %v2
; Z13-NEXT: vlgvg	%r1, %v0, 1
; Z13-NEXT: vlgvg	%r0, %v0, 0

; CHECK-NEXT: stpq	%r0, 0(%r2)
; CHECK-NEXT: bcr	1{{[45]}}, %r0
; CHECK-NEXT: br	%r14
  %val = load fp128, ptr %src, align 8
  %add = fadd fp128 %val, %val
  store atomic fp128 %add, ptr %dst seq_cst, align 16
  ret void
}

define void @f2(ptr %dst, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __atomic_store@PLT
  %val = load fp128, ptr %src, align 8
  store atomic fp128 %val, ptr %dst seq_cst, align 8
  ret void
}

define void @f2_fpuse(ptr %dst, ptr %src) {
; CHECK-LABEL: f2_fpuse:
; CHECK:       # %bb.0:
; CHECK-NEXT:	stmg	%r14, %r15, 112(%r15)
; CHECK-NEXT:	.cfi_offset %r14, -48
; CHECK-NEXT:	.cfi_offset %r15, -40
; CHECK-NEXT:	aghi	%r15, -176
; CHECK-NEXT:	.cfi_def_cfa_offset 336
; CHECK-NEXT:	ld	%f0, 0(%r3)
; CHECK-NEXT:	ld	%f2, 8(%r3)

; BASE-NEXT:	lgr	%r3, %r2
; BASE-NEXT:	axbr	%f0, %f0

; Z13-NEXT:	axbr	%f0, %f0
; Z13-NEXT:	lgr	%r3, %r2

; CHECK-NEXT:	la	%r4, 160(%r15)
; CHECK-NEXT:	lghi	%r2, 16
; CHECK-NEXT:	lhi	%r5, 5
; CHECK-NEXT:	std	%f0, 160(%r15)
; CHECK-NEXT:	std	%f2, 168(%r15)
; CHECK-NEXT: brasl %r14, __atomic_store@PLT
  %val = load fp128, ptr %src, align 8
  %add = fadd fp128 %val, %val
  store atomic fp128 %add, ptr %dst seq_cst, align 8
  ret void
}
