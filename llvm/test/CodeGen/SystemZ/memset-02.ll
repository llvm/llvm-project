; Test memset in cases where the set value is a constant other than 0 and -1.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memset.p0i8.i32(i8 *nocapture, i8, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8 *nocapture, i8, i64, i32, i1) nounwind

; No bytes, i32 version.
define void @f1(i8 *%dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 0, i32 1, i1 false)
  ret void
}

; No bytes, i64 version.
define void @f2(i8 *%dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 0, i32 1, i1 false)
  ret void
}

; 1 byte, i32 version.
define void @f3(i8 *%dest) {
; CHECK-LABEL: f3:
; CHECK: mvi 0(%r2), 128
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 1, i32 1, i1 false)
  ret void
}

; 1 byte, i64 version.
define void @f4(i8 *%dest) {
; CHECK-LABEL: f4:
; CHECK: mvi 0(%r2), 128
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 1, i32 1, i1 false)
  ret void
}

; 2 bytes, i32 version.
define void @f5(i8 *%dest) {
; CHECK-LABEL: f5:
; CHECK: mvhhi 0(%r2), -32640
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 2, i32 1, i1 false)
  ret void
}

; 2 bytes, i64 version.
define void @f6(i8 *%dest) {
; CHECK-LABEL: f6:
; CHECK: mvhhi 0(%r2), -32640
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 2, i32 1, i1 false)
  ret void
}

; 3 bytes, i32 version.
define void @f7(i8 *%dest) {
; CHECK-LABEL: f7:
; CHECK-DAG: mvhhi 0(%r2), -32640
; CHECK-DAG: mvi 2(%r2), 128
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 3, i32 1, i1 false)
  ret void
}

; 3 bytes, i64 version.
define void @f8(i8 *%dest) {
; CHECK-LABEL: f8:
; CHECK-DAG: mvhhi 0(%r2), -32640
; CHECK-DAG: mvi 2(%r2), 128
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 3, i32 1, i1 false)
  ret void
}

; 4 bytes, i32 version.
define void @f9(i8 *%dest) {
; CHECK-LABEL: f9:
; CHECK: iilf [[REG:%r[0-5]]], 2155905152
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 4, i32 1, i1 false)
  ret void
}

; 4 bytes, i64 version.
define void @f10(i8 *%dest) {
; CHECK-LABEL: f10:
; CHECK: iilf [[REG:%r[0-5]]], 2155905152
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 4, i32 1, i1 false)
  ret void
}

; 5 bytes, i32 version.
define void @f11(i8 *%dest) {
; CHECK-LABEL: f11:
; CHECK: mvi 0(%r2), 128
; CHECK: mvc 1(4,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 5, i32 1, i1 false)
  ret void
}

; 5 bytes, i64 version.
define void @f12(i8 *%dest) {
; CHECK-LABEL: f12:
; CHECK: mvi 0(%r2), 128
; CHECK: mvc 1(4,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 5, i32 1, i1 false)
  ret void
}

; 257 bytes, i32 version.
define void @f13(i8 *%dest) {
; CHECK-LABEL: f13:
; CHECK: mvi 0(%r2), 128
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 257, i32 1, i1 false)
  ret void
}

; 257 bytes, i64 version.
define void @f14(i8 *%dest) {
; CHECK-LABEL: f14:
; CHECK: mvi 0(%r2), 128
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 257, i32 1, i1 false)
  ret void
}

; 258 bytes, i32 version.  We need two MVCs.
define void @f15(i8 *%dest) {
; CHECK-LABEL: f15:
; CHECK: mvi 0(%r2), 128
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(1,%r2), 256(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 128, i32 258, i32 1, i1 false)
  ret void
}

; 258 bytes, i64 version.
define void @f16(i8 *%dest) {
; CHECK-LABEL: f16:
; CHECK: mvi 0(%r2), 128
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(1,%r2), 256(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 128, i64 258, i32 1, i1 false)
  ret void
}
