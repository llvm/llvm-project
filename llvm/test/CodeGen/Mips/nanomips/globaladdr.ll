; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs -mgpopt < %s | FileCheck %s --check-prefix=CHECK-GP

@str = constant [10 x i8] c"pineapple\00"

define i8* @foo() {
; CHECK: la $a0, str
; CHECK-GP: la $a0, str
  ret i8* getelementptr([10 x i8], [10 x i8]* @str, i64 0, i64 0)
}

@n = global i32 5

define i32 @load_value() {
; CHECK: la $a0, n
; CHECK: lw $a0, 0($a0)
; CHECK-GP: lw $a0, %gp_rel(n)($gp)
  %r = load i32, i32* @n
  ret i32 %r
}

define i32* @load_address() {
; CHECK: la $a0, n
; CHECK-GP: addiu $a0, $gp, %gp_rel(n)
  ret i32* @n
}
