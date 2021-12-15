; Test memset in cases where a loop is used.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memset.p0i8.i32(i8 *nocapture, i8, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8 *nocapture, i8, i64, i1) nounwind

; Constant length: 6 iterations and 2 bytes remainder.
define void @f1(i8* %dest, i8 %val) {
; CHECK-LABEL: f1:
; CHECK: lghi [[COUNT:%r[0-5]]], 6
; CHECK: [[LABEL:\.L[^:]*]]:
; CHECK: pfd 2, 768(%r2)
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(255,%r2), 0(%r2)
; CHECK: la %r2, 256(%r2)
; CHECK: brctg [[COUNT]], [[LABEL]]
; CHECK: stc %r3, 0(%r2)
; CHECK-NEXT: mvc 1(1,%r2), 0(%r2)
; CHECK-NEXT: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 1538, i1 false)
  ret void
}

; Constant length: 6 iterations and 255 bytes remainder.
define void @f2(i8* %dest) {
; CHECK-LABEL: f2:
; CHECK: lghi [[COUNT:%r[0-5]]], 6
; CHECK: [[LABEL:\.L[^:]*]]:
; CHECK: pfd 2, 768(%r2)
; CHECK: mvi  0(%r2), 1
; CHECK: mvc 1(255,%r2), 0(%r2)
; CHECK: la %r2, 256(%r2)
; CHECK: brctg [[COUNT]], [[LABEL]]
; CHECK: mvi  0(%r2), 1
; CHECK-NEXT: mvc 1(254,%r2), 0(%r2)
; CHECK-NEXT: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 1, i32 1791, i1 false)
  ret void
}

; Variable length, byte in register.
define void @f3(i8* %dest, i8 %val, i64 %Len) {
; CHECK-LABEL: f3:
; CHECK: # %bb.0:
; CHECK-NEXT: 	aghi	%r4, -2
; CHECK-NEXT: 	cgibe	%r4, -2, 0(%r14)
; CHECK-NEXT: .LBB2_1:
; CHECK-NEXT:	cgije	%r4, -1, .LBB2_5
; CHECK-NEXT:# %bb.2:
; CHECK-NEXT:	srlg	%r0, %r4, 8
; CHECK-NEXT:	cgije	%r0, 0, .LBB2_4
; CHECK-NEXT:.LBB2_3:                   # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:	pfd	2, 768(%r2)
; CHECK-NEXT:	stc	%r3, 0(%r2)
; CHECK-NEXT:	mvc	1(255,%r2), 0(%r2)
; CHECK-NEXT:	la	%r2, 256(%r2)
; CHECK-NEXT:	brctg	%r0, .LBB2_3
; CHECK-NEXT:.LBB2_4:
; CHECK-NEXT:	stc	%r3, 0(%r2)
; CHECK-NEXT:	exrl	%r4, .Ltmp0
; CHECK-NEXT:	br	%r14
; CHECK-NEXT:.LBB2_5:
; CHECK-NEXT:	stc	%r3, 0(%r2)
; CHECK-NEXT:	br	%r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 %Len, i1 false)
  ret void
}

; Variable length, immediate byte.
define void @f4(i8* %dest, i32 %Len) {
; CHECK-LABEL: f4:
; CHECK: # %bb.0:
; CHECK-NEXT:	llgfr	%r1, %r3
; CHECK-NEXT:	aghi	%r1, -2
; CHECK-NEXT:	cgibe	%r1, -2, 0(%r14)
; CHECK-NEXT:.LBB3_1:
; CHECK-NEXT:	cgije	%r1, -1, .LBB3_5
; CHECK-NEXT:# %bb.2:
; CHECK-NEXT:	srlg	%r0, %r1, 8
; CHECK-NEXT:	cgije	%r0, 0, .LBB3_4
; CHECK-NEXT:.LBB3_3:                   # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:	pfd	2, 768(%r2)
; CHECK-NEXT:	mvi	0(%r2), 1
; CHECK-NEXT:	mvc	1(255,%r2), 0(%r2)
; CHECK-NEXT:	la	%r2, 256(%r2)
; CHECK-NEXT:	brctg	%r0, .LBB3_3
; CHECK-NEXT:.LBB3_4:
; CHECK-NEXT:	mvi	0(%r2), 1
; CHECK-NEXT:	exrl	%r1, .Ltmp0
; CHECK-NEXT:	br	%r14
; CHECK-NEXT:.LBB3_5:
; CHECK-NEXT:	mvi	0(%r2), 1
; CHECK-NEXT:	br	%r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 1, i32 %Len, i1 false)
  ret void
}

; CHECK: .Ltmp0:
; CHECK-NEXT:	mvc	1(1,%r2), 0(%r2)
