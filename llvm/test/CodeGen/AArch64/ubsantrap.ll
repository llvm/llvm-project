; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios -global-isel %s -o - | FileCheck %s

define void @test_ubsantrap() {
; CHECK-LABEL: test_ubsantrap
; CHECK: brk #0x550c
  call void @llvm.ubsantrap(i8 12)
  ret void
}

define void @test_ubsantrap_function() {
; CHECK-LABEL: test_ubsantrap_function:
; CHECK: mov w0, #12
; CHECK: bl _wibble
  call void @llvm.ubsantrap(i8 12) "trap-func-name"="wibble"
  ret void
}

declare void @llvm.ubsantrap(i8)
