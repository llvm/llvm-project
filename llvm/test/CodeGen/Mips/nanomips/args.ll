; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=nanomips -verify-machineinstrs --stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ADJSTACK

declare i32 @ints9(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @test_ints9(i32 %a, i32 %b) {
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
; CHECK: balc ints9
; CHECK: BALC_NM
; ADJSTACK: ADJCALLSTACKDOWN_NM 16
; ADJSTACK: ADJCALLSTACKUP_NM 16
  %1 = call i32 @ints9(i32 %a, i32 %b, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  ret i32 %1
}

declare void @unaligned1(i32, i32, i32, i64, i32, i32, i32, i64)

; Make sure that $a3 register is skipped, because 64-bit values need to be in even registers.
define void @unaligned2(i32 %a, i32 %b) {
; CHECK-NOT: sw ${{[ats][0-9]}}, 4($sp)
; CHECK-NOT: li $a3
; CHECK: li ${{[ats][0-9]}}, 6
; CHECK: sw ${{[ats][0-9]}}, 8($sp)
; CHECK: li ${{[ats][0-9]}}, 5
; CHECK: sw ${{[ats][0-9]}}, 0($sp)
; CHECK: li $a5, 0
; CHECK: sw ${{[ats][0-9]}}, 12($sp)
; CHECK: li $a2, 1
; CHECK: li $a4, 2
; CHECK: li $a6, 3
; CHECK: li $a7, 4
  call void @unaligned1(i32 %a, i32 %b, i32 1, i64 2, i32 3, i32 4, i32 5, i64 6)
  ret void
}
