; Test memsets that set all bits.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memset.p0i8.i32(i8 *nocapture, i8, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8 *nocapture, i8, i64, i1) nounwind

; No bytes, i32 version.
define void @f1(i8* %dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 0, i1 false)
  ret void
}

; No bytes, i64 version.
define void @f2(i8* %dest) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 0, i1 false)
  ret void
}

; 1 byte, i32 version.
define void @f3(i8* %dest) {
; CHECK-LABEL: f3:
; CHECK: mvi 0(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 1, i1 false)
  ret void
}

; 1 byte, i64 version.
define void @f4(i8* %dest) {
; CHECK-LABEL: f4:
; CHECK: mvi 0(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 1, i1 false)
  ret void
}

; 2 bytes, i32 version.
define void @f5(i8* %dest) {
; CHECK-LABEL: f5:
; CHECK: mvhhi 0(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 2, i1 false)
  ret void
}

; 2 bytes, i64 version.
define void @f6(i8* %dest) {
; CHECK-LABEL: f6:
; CHECK: mvhhi 0(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 2, i1 false)
  ret void
}

; 3 bytes, i32 version.
define void @f7(i8* %dest) {
; CHECK-LABEL: f7:
; CHECK-DAG: mvhhi 0(%r2), -1
; CHECK-DAG: mvi 2(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 3, i1 false)
  ret void
}

; 3 bytes, i64 version.
define void @f8(i8* %dest) {
; CHECK-LABEL: f8:
; CHECK-DAG: mvhhi 0(%r2), -1
; CHECK-DAG: mvi 2(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 3, i1 false)
  ret void
}

; 4 bytes, i32 version.
define void @f9(i8* %dest) {
; CHECK-LABEL: f9:
; CHECK: mvhi 0(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 4, i1 false)
  ret void
}

; 4 bytes, i64 version.
define void @f10(i8* %dest) {
; CHECK-LABEL: f10:
; CHECK: mvhi 0(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 4, i1 false)
  ret void
}

; 5 bytes, i32 version.
define void @f11(i8* %dest) {
; CHECK-LABEL: f11:
; CHECK-DAG: mvhi 0(%r2), -1
; CHECK-DAG: mvi 4(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 5, i1 false)
  ret void
}

; 5 bytes, i64 version.
define void @f12(i8* %dest) {
; CHECK-LABEL: f12:
; CHECK-DAG: mvhi 0(%r2), -1
; CHECK-DAG: mvi 4(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 5, i1 false)
  ret void
}

; 6 bytes, i32 version.
define void @f13(i8* %dest) {
; CHECK-LABEL: f13:
; CHECK-DAG: mvhi 0(%r2), -1
; CHECK-DAG: mvhhi 4(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 6, i1 false)
  ret void
}

; 6 bytes, i64 version.
define void @f14(i8* %dest) {
; CHECK-LABEL: f14:
; CHECK-DAG: mvhi 0(%r2), -1
; CHECK-DAG: mvhhi 4(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 6, i1 false)
  ret void
}

; 7 bytes, i32 version.
define void @f15(i8* %dest) {
; CHECK-LABEL: f15:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(6,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 7, i1 false)
  ret void
}

; 7 bytes, i64 version.
define void @f16(i8* %dest) {
; CHECK-LABEL: f16:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(6,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 7, i1 false)
  ret void
}

; 8 bytes, i32 version.
define void @f17(i8* %dest) {
; CHECK-LABEL: f17:
; CHECK: mvghi 0(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 8, i1 false)
  ret void
}

; 8 bytes, i64 version.
define void @f18(i8* %dest) {
; CHECK-LABEL: f18:
; CHECK: mvghi 0(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 8, i1 false)
  ret void
}

; 9 bytes, i32 version.
define void @f19(i8* %dest) {
; CHECK-LABEL: f19:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvi 8(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 9, i1 false)
  ret void
}

; 9 bytes, i64 version.
define void @f20(i8* %dest) {
; CHECK-LABEL: f20:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvi 8(%r2), 255
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 9, i1 false)
  ret void
}

; 10 bytes, i32 version.
define void @f21(i8* %dest) {
; CHECK-LABEL: f21:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvhhi 8(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 10, i1 false)
  ret void
}

; 10 bytes, i64 version.
define void @f22(i8* %dest) {
; CHECK-LABEL: f22:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvhhi 8(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 10, i1 false)
  ret void
}

; 11 bytes, i32 version.
define void @f23(i8* %dest) {
; CHECK-LABEL: f23:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(10,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 11, i1 false)
  ret void
}

; 11 bytes, i64 version.
define void @f24(i8* %dest) {
; CHECK-LABEL: f24:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(10,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 11, i1 false)
  ret void
}

; 12 bytes, i32 version.
define void @f25(i8* %dest) {
; CHECK-LABEL: f25:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvhi 8(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 12, i1 false)
  ret void
}

; 12 bytes, i64 version.
define void @f26(i8* %dest) {
; CHECK-LABEL: f26:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvhi 8(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 12, i1 false)
  ret void
}

; 13 bytes, i32 version.
define void @f27(i8* %dest) {
; CHECK-LABEL: f27:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(12,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 13, i1 false)
  ret void
}

; 13 bytes, i64 version.
define void @f28(i8* %dest) {
; CHECK-LABEL: f28:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(12,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 13, i1 false)
  ret void
}

; 14 bytes, i32 version.
define void @f29(i8* %dest) {
; CHECK-LABEL: f29:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(13,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 14, i1 false)
  ret void
}

; 14 bytes, i64 version.
define void @f30(i8* %dest) {
; CHECK-LABEL: f30:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(13,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 14, i1 false)
  ret void
}

; 15 bytes, i32 version.
define void @f31(i8* %dest) {
; CHECK-LABEL: f31:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(14,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 15, i1 false)
  ret void
}

; 15 bytes, i64 version.
define void @f32(i8* %dest) {
; CHECK-LABEL: f32:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(14,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 15, i1 false)
  ret void
}

; 16 bytes, i32 version.
define void @f33(i8* %dest) {
; CHECK-LABEL: f33:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvghi 8(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 16, i1 false)
  ret void
}

; 16 bytes, i64 version.
define void @f34(i8* %dest) {
; CHECK-LABEL: f34:
; CHECK-DAG: mvghi 0(%r2), -1
; CHECK-DAG: mvghi 8(%r2), -1
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 16, i1 false)
  ret void
}

; 17 bytes, i32 version.
define void @f35(i8* %dest) {
; CHECK-LABEL: f35:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(16,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 17, i1 false)
  ret void
}

; 17 bytes, i64 version.
define void @f36(i8* %dest) {
; CHECK-LABEL: f36:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(16,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 17, i1 false)
  ret void
}

; 257 bytes, i32 version.
define void @f37(i8* %dest) {
; CHECK-LABEL: f37:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 257, i1 false)
  ret void
}

; 257 bytes, i64 version.
define void @f38(i8* %dest) {
; CHECK-LABEL: f38:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 257, i1 false)
  ret void
}

; 258 bytes, i32 version.  We need two MVCs.
define void @f39(i8* %dest) {
; CHECK-LABEL: f39:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(1,%r2), 256(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 -1, i32 258, i1 false)
  ret void
}

; 258 bytes, i64 version.
define void @f40(i8* %dest) {
; CHECK-LABEL: f40:
; CHECK: mvi 0(%r2), 255
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: mvc 257(1,%r2), 256(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 -1, i64 258, i1 false)
  ret void
}
