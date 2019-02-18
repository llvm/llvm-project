; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+xsave | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xsave | FileCheck %s --check-prefix=CHECK64

define void @test_xsetbv(i32 %in, i32 %high, i32 %low) {
; CHECK-LABEL: test_xsetbv
; CHECK: movl  4(%esp), %ecx
; CHECK: movl  8(%esp), %edx
; CHECK: movl  12(%esp), %eax
; CHECK: xsetbv
; CHECK: ret

; CHECK64-LABEL: test_xsetbv
; CHECK64: movl  %edx, %eax
; CHECK64-DAG: movl  %edi, %ecx
; CHECK64-DAG: movl  %esi, %edx
; CHECK64: xsetbv
; CHECK64: ret

  call void @llvm.x86.xsetbv(i32 %in, i32 %high, i32 %low)
  ret void;
}
declare void @llvm.x86.xsetbv(i32, i32, i32)

