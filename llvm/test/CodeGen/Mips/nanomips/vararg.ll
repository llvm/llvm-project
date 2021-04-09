; RUN: llc -mtriple=nanomips -asm-show-inst < %s | FileCheck %s

declare i32 @varfunc(i32, ...)

define i32 @test_varargs() {
; CHECK: addiu $sp, $sp, -16
; CHECK: ADDiu_NM
; CHECK: sw $ra, 12($sp)
; CHECK: SW_NM
; CHECK: li $a0, 3
; CHECK: Li_NM
; CHECK: li $a1, 25
; CHECK: Li_NM
; CHECK: li $a2, 24
; CHECK: Li_NM
; CHECK: li $a3, 23
; CHECK: Li_NM
; CHECK: balc varfunc
; CHECK: BALC_NM
  %1 = call i32 (i32, ...) @varfunc(i32 3, i32 25, i32 24, i32 23)
; CHECK: lw $ra, 12($sp)
; CHECK: LW_NM
; CHECK: addiu $sp, $sp, 16
; CHECK: ADDiu_NM
  ret i32 %1
}
