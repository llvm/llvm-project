
; RUN: opt -passes=instsimplify -S < %s | FileCheck %s

declare i8 @llvm.ctpop.i8(i8)

define i1 @ctpop_ule_operand(i8 %x) {
; CHECK-LABEL: @ctpop_ule_operand(
; CHECK: ret i1 true

  %pop = call i8 @llvm.ctpop.i8(i8 %x)
  %cmp = icmp ule i8 %pop, %x
  ret i1 %cmp
}

