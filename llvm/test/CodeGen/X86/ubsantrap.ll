; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s

define void @test_ubsantrap() {
; CHECK-LABEL: test_ubsantrap
; CHECK: ud1l 12(%eax), %eax
  call void @llvm.ubsantrap(i8 12)
  ret void
}

define void @test_ubsantrap_function() {
; CHECK-LABEL: test_ubsantrap_function:
; CHECK: movl $12, %edi
; CHECK: callq wibble
  call void @llvm.ubsantrap(i8 12) "trap-func-name"="wibble"
  ret void
}

declare void @llvm.ubsantrap(i8)
