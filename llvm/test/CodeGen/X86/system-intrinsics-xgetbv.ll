; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+xsave | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xsave | FileCheck %s --check-prefix=CHECK64

define i64 @test_xgetbv(i32 %in) {
; CHECK-LABEL: test_xgetbv
; CHECK: movl  4(%esp), %ecx
; CHECK: xgetbv
; CHECK: ret

; CHECK64-LABEL: test_xgetbv
; CHECK64: movl  %edi, %ecx
; CHECK64: xgetbv
; CHECK64: shlq  $32, %rdx
; CHECK64: orq   %rdx, %rax
; CHECK64: ret

  %1 = call i64 @llvm.x86.xgetbv(i32 %in)
  ret i64 %1;
}

declare i64 @llvm.x86.xgetbv(i32)