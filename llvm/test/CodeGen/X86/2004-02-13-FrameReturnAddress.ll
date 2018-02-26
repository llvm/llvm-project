; RUN: llc < %s -mtriple=i686-- | FileCheck %s

declare i8* @llvm.returnaddress(i32)

declare i8* @llvm.frameaddress(i32)

define i8* @test1() {
; CHECK-LABEL: test1:
entry:
  %X = call i8* @llvm.returnaddress( i32 0 )
  ret i8* %X
; CHECK: movl {{.*}}(%esp), %eax
}

define i8* @test2() {
; CHECK-LABEL: test2:
entry:
  %X = call i8* @llvm.frameaddress( i32 0 )
  ret i8* %X
; CHECK: pushl %ebp
; CHECK: popl %ebp
}

