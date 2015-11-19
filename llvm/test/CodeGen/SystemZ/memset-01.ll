; Test memset in cases where the set value is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memset.p0i8.i32(i8 *nocapture, i8, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8 *nocapture, i8, i64, i1) nounwind

; No bytes, i32 version.
define void @f1(i8 *%dest, i8 %val) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 0, i1 false)
  ret void
}

; No bytes, i64 version.
define void @f2(i8 *%dest, i8 %val) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 0, i1 false)
  ret void
}

; 1 byte, i32 version.
define void @f3(i8 *%dest, i8 %val) {
; CHECK-LABEL: f3:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 1, i1 false)
  ret void
}

; 1 byte, i64 version.
define void @f4(i8 *%dest, i8 %val) {
; CHECK-LABEL: f4:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 1, i1 false)
  ret void
}

; 2 bytes, i32 version.
define void @f5(i8 *%dest, i8 %val) {
; CHECK-LABEL: f5:
; CHECK-DAG: stc %r3, 0(%r2)
; CHECK-DAG: stc %r3, 1(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 2, i1 false)
  ret void
}

; 2 bytes, i64 version.
define void @f6(i8 *%dest, i8 %val) {
; CHECK-LABEL: f6:
; CHECK-DAG: stc %r3, 0(%r2)
; CHECK-DAG: stc %r3, 1(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 2, i1 false)
  ret void
}

; 3 bytes, i32 version.
define void @f7(i8 *%dest, i8 %val) {
; CHECK-LABEL: f7:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(2,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 3, i1 false)
  ret void
}

; 3 bytes, i64 version.
define void @f8(i8 *%dest, i8 %val) {
; CHECK-LABEL: f8:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(2,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 3, i1 false)
  ret void
}

; 257 bytes, i32 version.
define void @f9(i8 *%dest, i8 %val) {
; CHECK-LABEL: f9:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 257, i1 false)
  ret void
}

; 257 bytes, i64 version.
define void @f10(i8 *%dest, i8 %val) {
; CHECK-LABEL: f10:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 257, i1 false)
  ret void
}

; 258 bytes, i32 version.  We need two MVCs.
define void @f11(i8 *%dest, i8 %val) {
; CHECK-LABEL: f11:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(1,%r2), 256(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 258, i1 false)
  ret void
}

; 258 bytes, i64 version.
define void @f12(i8 *%dest, i8 %val) {
; CHECK-LABEL: f12:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(1,%r2), 256(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 258, i1 false)
  ret void
}

; Test the largest case for which straight-line code is used.
define void @f13(i8 *%dest, i8 %val) {
; CHECK-LABEL: f13:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(256,%r2), 256(%r2)
; CHECK: mvc 513(256,%r2), 512(%r2)
; CHECK: mvc 769(256,%r2), 768(%r2)
; CHECK: mvc 1025(256,%r2), 1024(%r2)
; CHECK: mvc 1281(256,%r2), 1280(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 1537,
                                  i1 false)
  ret void
}

; Test the next size up, which uses a loop.  We leave the other corner
; cases to memcpy-01.ll.
define void @f14(i8 *%dest, i8 %val) {
; CHECK-LABEL: f14:
; CHECK: stc %r3, 0(%r2)
; CHECK: lghi [[COUNT:%r[0-5]]], 6
; CHECK: [[LABEL:\.L[^:]*]]:
; CHECK: pfd 2, 769(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: la %r2, 256(%r2)
; CHECK: brctg [[COUNT]], [[LABEL]]
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 1538,
                                  i1 false)
  ret void
}
