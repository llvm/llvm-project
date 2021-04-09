; RUN: llc -mtriple=nanomips -asm-show-inst < %s | FileCheck %s

define i32 @foo(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  ret i32 %1
}

define i32 @bar(i32 %a, i32 %b) {
; CHECK: addiu $sp, $sp, -16
; CHECK: ADDiu_NM
; CHECK: sw $ra, 12($sp)
; CHECK: SW_NM
; CHECK: balc foo
; CHECK: BALC_NM
  %1 = call i32 @foo(i32 %a, i32 %b)
; CHECK: lw $ra, 12($sp)
; CHECK: LW
; CHECK: addiu $sp, $sp, 16
; CHECK: ADDiu_NM
  ret i32 %1
}
