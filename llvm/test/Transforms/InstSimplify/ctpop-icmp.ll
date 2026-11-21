
; RUN: opt -passes=instsimplify -S < %s | FileCheck %s

declare i8 @llvm.ctpop.i8(i8)

define i1 @ctpop_ule_operand_i8(i8 %x) {
; CHECK-LABEL: @ctpop_ule_operand_i8(
; CHECK: ret i1 true

  %pop = call i8 @llvm.ctpop.i8(i8 %x)
  %cmp = icmp ule i8 %pop, %x
  ret i1 %cmp
}


declare i32 @llvm.ctpop.i32(i32)

define i1 @ctpop_ule_operand_i32(i32 %x){
; CHECK-LABEL: @ctpop_ule_operand_i32(
; CHECK: ret i1 true

  %pop = call i32 @llvm.ctpop.i32(i32 %x)
  %cmp = icmp ule i32 %pop, %x
  ret i1 %cmp

}

declare i64 @llvm.ctpop.i64(i64)

define i1 @ctpop_ule_operand_i64(i64 %x){
; CHECK-LABEL: @ctpop_ule_operand_i64(
; CHECK: ret i1 true

  %pop = call i64 @llvm.ctpop.i64(i64 %x)
  %cmp = icmp ule i64 %pop, %x
  ret i1 %cmp

}


define i1 @ctpop_ugt_operand_i8(i8 %x) {
; CHECK-LABEL: @ctpop_ugt_operand_i8(
; CHECK: ret i1 false

  %pop = call i8 @llvm.ctpop.i8(i8 %x)
  %cmp = icmp ugt i8 %pop, %x
  ret i1 %cmp
}

define i1 @ctpop_ugt_operand_i32(i32 %x) {
; CHECK-LABEL: @ctpop_ugt_operand_i32(
; CHECK: ret i1 false

  %pop = call i32 @llvm.ctpop.i32(i32 %x)
  %cmp = icmp ugt i32 %pop, %x
  ret i1 %cmp
}

define i1 @ctpop_ugt_operand_i64(i64 %x) {
; CHECK-LABEL: @ctpop_ugt_operand_i64(
; CHECK: ret i1 false

  %pop = call i64 @llvm.ctpop.i64(i64 %x)
  %cmp = icmp ugt i64 %pop, %x
  ret i1 %cmp
}
