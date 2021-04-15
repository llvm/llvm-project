; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

@str = constant [10 x i8] c"pineapple\00"

define i8* @foo() {
; CHECK: li $a0
; CHECK: Li_NM
  ret i8* getelementptr([10 x i8], [10 x i8]* @str, i64 0, i64 0)
}
