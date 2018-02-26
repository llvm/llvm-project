; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Allocate 8 bytes, no need to align stack.
define void @f0() {
; CHECK-LABEL: f0:
; CHECK: aghi %r15, -168
; CHECK-NOT: nil
; CHECK: mvghi 160(%r15), 10
; CHECK: aghi %r15, 168
  %x = alloca i64
  store volatile i64 10, i64* %x
  ret void
}

; Allocate %len * 8, no need to align stack.
define void @f1(i64 %len) {
; CHECK-LABEL: f1:
; CHECK-DAG: sllg    %r0, %r2, 3
; CHECK-DAG: lgr     %r1, %r15
; CHECK: sgr     %r1, %r0
; CHECK-NOT: ngr
; CHECK-DAG: lgr     %r15, %r1
; CHECK-DAG: la      %r2, 160(%r1)
; CHECK: mvghi   0(%r2), 10
  %x = alloca i64, i64 %len
  store volatile i64 10, i64* %x
  ret void
}

; Static alloca, align 128.
define void @f2() {
; CHECK-LABEL: f2:
; CHECK: aghi    %r1, -128
; CHECK-DAG: lgr     %r15, %r1
; CHECK-DAG: la      %r2, 280(%r1)
; CHECK-DAG: nill    %r2, 65408
; CHECK: mvghi   0(%r2), 10
  %x = alloca i64, i64 1, align 128
  store volatile i64 10, i64* %x, align 128
  ret void
}

; Dynamic alloca, align 128.
define void @f3(i64 %len) {
; CHECK-LABEL: f3:
; CHECK-DAG: sllg	%r2, %r2, 3
; CHECK-DAG: la	%r0, 120(%r2)
; CHECK-DAG: lgr	%r1, %r15
; CHECK: sgr	%r1, %r0
; CHECK: la	%r2, 280(%r1)
; CHECK: nill	%r2, 65408
; CHECK: lgr	%r15, %r1
; CHECK: mvghi	0(%r2), 10
  %x = alloca i64, i64 %len, align 128
  store volatile i64 10, i64* %x, align 128
  ret void
}

; Static alloca w/out alignment - part of frame.
define void @f4() {
; CHECK-LABEL: f4:
; CHECK: aghi    %r15, -168
; CHECK: mvhi    164(%r15), 10
; CHECK: aghi    %r15, 168
  %x = alloca i32
  store volatile i32 10, i32* %x
  ret void
}

; Static alloca of one i32, aligned by 128.
define void @f5() {
; CHECK-LABEL: f5:

; CHECK: lgr	%r1, %r15
; CHECK: aghi	%r1, -128
; CHECK: la	%r2, 280(%r1)
; CHECK: nill	%r2, 65408
; CHECK: lgr	%r15, %r1
; CHECK: mvhi	0(%r2), 10
  %x = alloca i32, i64 1, align 128
  store volatile i32 10, i32* %x
  ret void
}

