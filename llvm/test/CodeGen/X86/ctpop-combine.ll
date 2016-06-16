; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define i32 @test1(i64 %x) nounwind readnone {
  %count = tail call i64 @llvm.ctpop.i64(i64 %x)
  %cast = trunc i64 %count to i32
  %cmp = icmp ugt i32 %cast, 1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: test1:
; CHECK: leaq -1([[A0:%rdi|%rcx]])
; CHECK-NEXT: testq
; CHECK-NEXT: setne
; CHECK: ret
}


define i32 @test2(i64 %x) nounwind readnone {
  %count = tail call i64 @llvm.ctpop.i64(i64 %x)
  %cmp = icmp ult i64 %count, 2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: test2:
; CHECK: leaq -1([[A0]])
; CHECK-NEXT: testq
; CHECK-NEXT: sete
; CHECK: ret
}

define i32 @test3(i64 %x) nounwind readnone {
  %count = tail call i64 @llvm.ctpop.i64(i64 %x)
  %cast = trunc i64 %count to i6 ; Too small for 0-64
  %cmp = icmp ult i6 %cast, 2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: test3:
; CHECK: cmpb $2
; CHECK: ret
}
