; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; Make sure that SAVE/RESTORE instructions are used for saving and restoring callee-saved registers.
define void @test() {
; CHECK: save 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
  call void asm sideeffect "", "~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$23},~{$22},~{$1}"() ret void
; CHECK: restore.jrc 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
}

; Make sure that SAVE/RESTORE instructions are not used when the offset is larger than 4092.
define void @test2() {
; CHECK-NOT: save
  %foo = alloca [4096 x i8], align 1
  %1 = getelementptr inbounds [4096 x i8], [4096 x i8]* %foo, i32 0, i32 0
  call void asm sideeffect "", "r,~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$23},~{$22},~{$1}"(i8* %1)
  ret void
; CHECK-NOT: restore.jrc
; CHECK-NOT: restore
}

; Make sure that SAVE/SAVE combination is used when incoming arguments need to
; be stored on the stack. First SAVE to move the stack pointer to where s-regs
; need to be stored and second SAVE to actually save the registers. Same logic
; applies to RESTORE, but inversed.
define void @test3(i32 %n0, i32 %n1, i32 %n2, i32 %n3, ...) {
; CHECK: save 16
; CHECK: save 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
  call void asm sideeffect "", "~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$23},~{$22},~{$1}"()
  ret void
; CHECK: restore 32, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK: restore.jrc 16
}

; Make sure to generate SAVE/RESTORE instruction in case when the first register
; is stored to the offset that is not doubleword aligned. In such case, it is
; neccessary to generate SAVE/RESTORE without the incompatible register and
; separate SW/LW for it.
define void @test4(i32 %n, ...) {
; CHECK: save 32
; CHECK: save 32, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK: sw $s0, 32($sp)
  call void asm sideeffect "", "~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$23},~{$22},~{$1}"()
  ret void
; CHECK: lw $s0, 32($sp)
; CHECK: restore 32, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK: restore.jrc 32
}

; Make sure to not generate SAVE/RESTORE without register arguments when the
; offset is not quadword aligned. Instead, generate addiu.
define void @test5(i32 %n0, i32 %n1, i32 %n2, i32 %n3, i32 %n4, i32 %n5, ...) {
; CHECK: addiu $sp, $sp, -8
; CHECK: save 40, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
  call void asm sideeffect "", "~{$16},~{$17},~{$18},~{$19},~{$20},~{$21},~{$23},~{$22},~{$1}"()
  ret void
; CHECK: restore 40, $s0, $s1, $s2, $s3, $s4, $s5, $s6, $s7
; CHECK: addiu $sp, $sp, 8
; CHECK: jrc $ra
}
