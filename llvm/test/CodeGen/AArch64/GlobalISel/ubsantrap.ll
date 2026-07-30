; RUN: llc -mtriple=arm64-apple-ios %s -o - -global-isel -global-isel-abort=1 | FileCheck %s

define void @test_ubsantrap() {
; CHECK-LABEL: test_ubsantrap
; CHECK: brk #0x550c
; CHECK-GISEL: brk #0x550c
  call void @llvm.ubsantrap(i8 12)
  ret void
}

declare void @llvm.ubsantrap(i8)
