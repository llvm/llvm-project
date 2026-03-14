; Test floating-point control register intrinsics.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.s390.sfpc(i32)
declare i32 @llvm.s390.efpc()

; SFPC.
define void @test_sfpc(i32 %fpc) {
; CHECK-LABEL: test_sfpc:
; CHECK: sfpc %r2
; CHECK: br %r14
  call void @llvm.s390.sfpc(i32 %fpc)
  ret void
}

; EFPC.
define i32 @test_efpc() {
; CHECK-LABEL: test_efpc:
; CHECK: efpc %r2
; CHECK: br %r14
  %res = call i32 @llvm.s390.efpc()
  ret i32 %res
}

; LFPC.
define void @test_lfpc1(ptr %ptr) {
; CHECK-LABEL: test_lfpc1:
; CHECK: lfpc 0(%r2)
; CHECK: br %r14
  %fpc = load i32, ptr %ptr
  call void @llvm.s390.sfpc(i32 %fpc)
  ret void
}

; LFPC with offset.
define void @test_lfpc2(ptr %ptr) {
; CHECK-LABEL: test_lfpc2:
; CHECK: lfpc 4092(%r2)
; CHECK: br %r14
  %ptr1 = getelementptr i32, ptr %ptr, i32 1023
  %fpc = load i32, ptr %ptr1
  call void @llvm.s390.sfpc(i32 %fpc)
  ret void
}

; STFPC.
define void @test_stfpc1(ptr %ptr) {
; CHECK-LABEL: test_stfpc1:
; CHECK: stfpc 0(%r2)
; CHECK: br %r14
  %fpc = call i32 @llvm.s390.efpc()
  store i32 %fpc, ptr %ptr
  ret void
}

; STFPC with offset.
define void @test_stfpc2(ptr %ptr) {
; CHECK-LABEL: test_stfpc2:
; CHECK: stfpc 4092(%r2)
; CHECK: br %r14
  %fpc = call i32 @llvm.s390.efpc()
  %ptr1 = getelementptr i32, ptr %ptr, i32 1023
  store i32 %fpc, ptr %ptr1
  ret void
}

