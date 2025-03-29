; Test long double atomic stores - via i128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck -check-prefixes=CHECK,BASE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck -check-prefixes=CHECK,Z13 %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=+soft-float | FileCheck -check-prefixes=SOFTFP %s

define void @f1(ptr %dst, ptr %src) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lg %r1, 8(%r3)
; CHECK-NEXT:    lg %r0, 0(%r3)
; CHECK-NEXT:    stpq %r0, 0(%r2)
; CHECK-NEXT:    bcr 1{{[45]}}, %r0
; CHECK-NEXT:    br %r14

; SOFTFP-LABEL: f1:
; SOFTFP:       # %bb.0:
; SOFTFP-NEXT:    lg %r1, 8(%r3)
; SOFTFP-NEXT:    lg %r0, 0(%r3)
; SOFTFP-NEXT:    stpq %r0, 0(%r2)
; SOFTFP-NEXT:    bcr 1{{[45]}}, %r0
; SOFTFP-NEXT:    br %r14
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

; SOFTFP-LABEL: f1_fpsrc:
; SOFTFP: lg	%r0, 8(%r3)
; SOFTFP-NEXT: lg	%r1, 0(%r3)
; SOFTFP-NEXT:	lgr	%r13, %r2
; SOFTFP-NEXT:	stg	%r0, 168(%r15)
; SOFTFP-NEXT:	stg	%r1, 160(%r15)
; SOFTFP-NEXT:	stg	%r0, 184(%r15)
; SOFTFP-NEXT:	la	%r2, 192(%r15)
; SOFTFP-NEXT:	la	%r3, 176(%r15)
; SOFTFP-NEXT:	la	%r4, 160(%r15)
; SOFTFP-NEXT:	stg	%r1, 176(%r15)
; SOFTFP-NEXT:	brasl	%r14, __addtf3@PLT
; SOFTFP-NEXT:	lg	%r1, 200(%r15)
; SOFTFP-NEXT:	lg	%r0, 192(%r15)
; SOFTFP-NEXT:	stpq	%r0, 0(%r13)
; SOFTFP-NEXT:	bcr	1{{[45]}}, %r0
; SOFTFP-NEXT:	lmg	%r13, %r15, 312(%r15)
; SOFTFP-NEXT:	br	%r14

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
; CHECK-DAG:	lgr	%r3, %r2
; CHECK-DAG:	axbr	%f0, %f0
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
