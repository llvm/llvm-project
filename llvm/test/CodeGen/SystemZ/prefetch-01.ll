; Test data prefetching.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.prefetch(i8*, i32, i32, i32)

@g = dso_local global [4096 x i8] zeroinitializer

; Check that instruction read prefetches are ignored.
define dso_local void @f1(i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: br %r14
  call void @llvm.prefetch(i8 *%ptr, i32 0, i32 0, i32 0)
  ret void
}

; Check that instruction write prefetches are ignored.
define dso_local void @f2(i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: br %r14
  call void @llvm.prefetch(i8 *%ptr, i32 1, i32 0, i32 0)
  ret void
}

; Check data read prefetches.
define dso_local void @f3(i8 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: pfd 1, 0(%r2)
; CHECK: br %r14
  call void @llvm.prefetch(i8 *%ptr, i32 0, i32 0, i32 1)
  ret void
}

; Check data write prefetches.
define dso_local void @f4(i8 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: pfd 2, 0(%r2)
; CHECK: br %r14
  call void @llvm.prefetch(i8 *%ptr, i32 1, i32 0, i32 1)
  ret void
}

; Check an address at the negative end of the range.
define dso_local void @f5(i8 *%base, i64 %index) {
; CHECK-LABEL: f5:
; CHECK: pfd 2, -524288({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %add = add i64 %index, -524288
  %ptr = getelementptr i8, i8 *%base, i64 %add
  call void @llvm.prefetch(i8 *%ptr, i32 1, i32 0, i32 1)
  ret void
}

; Check an address at the positive end of the range.
define dso_local void @f6(i8 *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: pfd 2, 524287({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %add = add i64 %index, 524287
  %ptr = getelementptr i8, i8 *%base, i64 %add
  call void @llvm.prefetch(i8 *%ptr, i32 1, i32 0, i32 1)
  ret void
}

; Check that the next address up still compiles.
define dso_local void @f7(i8 *%base, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: 524288
; CHECK: pfd 2,
; CHECK: br %r14
  %add = add i64 %index, 524288
  %ptr = getelementptr i8, i8 *%base, i64 %add
  call void @llvm.prefetch(i8 *%ptr, i32 1, i32 0, i32 1)
  ret void
}

; Check pc-relative prefetches.
define dso_local void @f8() {
; CHECK-LABEL: f8:
; CHECK: pfdrl 2, g
; CHECK: br %r14
  %ptr = getelementptr [4096 x i8], [4096 x i8] *@g, i64 0, i64 0
  call void @llvm.prefetch(i8 *%ptr, i32 1, i32 0, i32 1)
  ret void
}
