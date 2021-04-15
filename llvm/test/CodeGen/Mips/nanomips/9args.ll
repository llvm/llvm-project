; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

declare i32 @foo(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @bar(i32 %a, i32 %b) {
; CHECK: addiu $sp, $sp, -32
; CHECK: ADDiu_NM
; CHECK: sw $ra, 28($sp)
; CHECK: SW_NM
; CHECK: li ${{[ats][0-9]}}, 7
; CHECK: Li_NM
; CHECK: sw ${{[ats][0-9]}}, 0($sp)
; CHECK: SW_NM
; CHECK: li $a2, 1
; CHECK: Li_NM
; CHECK: li $a3, 2
; CHECK: Li_NM
; CHECK: li $a4, 3
; CHECK: Li_NM
; CHECK: li $a5, 4
; CHECK: Li_NM
; CHECK: li $a6, 5
; CHECK: Li_NM
; CHECK: li $a7, 6
; CHECK: Li_NM
; CHECK: balc foo
; CHECK: BALC_NM
  %1 = call i32 @foo(i32 %a, i32 %b, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  ret i32 %1
}
