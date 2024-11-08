; Test long double atomic loads - via i128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck -check-prefixes=CHECK,BASE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck -check-prefixes=CHECK,Z13 %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=+soft-float | FileCheck -check-prefixes=SOFTFP %s

define void @f1(ptr %ret, ptr %src) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lpq %r0, 0(%r3)
; CHECK-NEXT:    stg %r1, 8(%r2)
; CHECK-NEXT:    stg %r0, 0(%r2)
; CHECK-NEXT:    br %r14

; SOFTFP-LABEL: f1:
; SOFTFP:       # %bb.0:
; SOFTFP-NEXT:    lpq %r0, 0(%r3)
; SOFTFP-NEXT:    stg %r1, 8(%r2)
; SOFTFP-NEXT:    stg %r0, 0(%r2)
; SOFTFP-NEXT:    br %r14
  %val = load atomic fp128, ptr %src seq_cst, align 16
  store fp128 %val, ptr %ret, align 8
  ret void
}

define void @f1_fpuse(ptr %ret, ptr %src) {
; CHECK-LABEL: f1_fpuse:
; CHECK:       # %bb.0:
; CHECK-NEXT:	lpq	%r0, 0(%r3)

; BASE-NEXT: ldgr	%f0, %r0
; BASE-NEXT: ldgr	%f2, %r1

; Z13-NEXT: vlvgp %v0, %r0, %r1
; Z13-NEXT: vrepg %v2, %v0, 1

; CHECK-NEXT:	axbr	%f0, %f0
; CHECK-NEXT:	std	%f0, 0(%r2)
; CHECK-NEXT:	std	%f2, 8(%r2)
; CHECK-NEXT:	br	%r14


; SOFTFP-LABEL: f1_fpuse:
; SOFTFP: stmg	%r13, %r15, 104(%r15)
; SOFTFP: aghi	%r15, -208
; SOFTFP:	lpq	%r0, 0(%r3)
; SOFTFP-NEXT: lgr	%r13, %r2
; SOFTFP-NEXT: stg	%r1, 168(%r15)
; SOFTFP-NEXT: stg	%r0, 160(%r15)
; SOFTFP-NEXT: stg	%r1, 184(%r15)
; SOFTFP-NEXT: la	%r2, 192(%r15)
; SOFTFP-NEXT: la	%r3, 176(%r15)
; SOFTFP-NEXT: la	%r4, 160(%r15)
; SOFTFP-NEXT: stg	%r0, 176(%r15)
; SOFTFP-NEXT: brasl	%r14, __addtf3@PLT
; SOFTFP-NEXT: lg	%r0, 200(%r15)
; SOFTFP-NEXT: lg	%r1, 192(%r15)
; SOFTFP-NEXT: stg	%r0, 8(%r13)
; SOFTFP-NEXT: stg	%r1, 0(%r13)
; SOFTFP-NEXT: lmg	%r13, %r15, 312(%r15)
; SOFTFP-NEXT: br	%r14
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
