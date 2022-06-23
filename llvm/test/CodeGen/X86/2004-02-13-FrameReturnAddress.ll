; RUN: llc < %s -mtriple=i686-- | FileCheck %s

declare ptr @llvm.returnaddress(i32)

declare ptr @llvm.frameaddress(i32)

define ptr @test1() {
; CHECK-LABEL: test1:
entry:
  %X = call ptr @llvm.returnaddress( i32 0 )
  ret ptr %X
; CHECK: movl {{.*}}(%esp), %eax
}

define ptr @test2() {
; CHECK-LABEL: test2:
entry:
  %X = call ptr @llvm.frameaddress( i32 0 )
  ret ptr %X
; CHECK: pushl %ebp
; CHECK: popl %ebp
}

