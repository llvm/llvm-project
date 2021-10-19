; RUN: llc -mcpu=mips32r2 -mtriple=mipsel -relocation-model=static -o - %s | FileCheck %s

define void @isr_sw0() #0 {
; CHECK-LABEL: isr_sw0:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, [[R1:[0-9]+]]($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, [[R2:[0-9]+]]($sp)
; CHECK: ins    $27, $zero, 8, 1
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ; Must save all registers
; CHECK: sw      $7, {{[0-9]+}}($sp)
; CHECK: sw      $6, {{[0-9]+}}($sp)
; CHECK: sw      $5, {{[0-9]+}}($sp)
; CHECK: sw      $4, {{[0-9]+}}($sp)
; CHECK: sw      $3, {{[0-9]+}}($sp)
; CHECK: sw      $2, {{[0-9]+}}($sp)
; CHECK: sw      $25, {{[0-9]+}}($sp)
; CHECK: sw      $24, {{[0-9]+}}($sp)
; CHECK: sw      $15, {{[0-9]+}}($sp)
; CHECK: sw      $14, {{[0-9]+}}($sp)
; CHECK: sw      $13, {{[0-9]+}}($sp)
; CHECK: sw      $12, {{[0-9]+}}($sp)
; CHECK: sw      $11, {{[0-9]+}}($sp)
; CHECK: sw      $10, {{[0-9]+}}($sp)
; CHECK: sw      $9, {{[0-9]+}}($sp)
; CHECK: sw      $8, {{[0-9]+}}($sp)
; CHECK: sw      $ra, [[R5:[0-9]+]]($sp)
; CHECK: sw      $gp, {{[0-9]+}}($sp)
; CHECK: sw      $1, {{[0-9]+}}($sp)
; CHECK: mflo    $26
; CHECK: sw      $26, [[R3:[0-9]+]]($sp)
; CHECK: mfhi    $26
; CHECK: sw      $26, [[R4:[0-9]+]]($sp)
  call void bitcast (void (...)* @write to void ()*)()
; CHECK: lw      $26, [[R4:[0-9]+]]($sp)
; CHECK: mthi    $26
; CHECK: lw      $26, [[R3:[0-9]+]]($sp)
; CHECK: mtlo    $26
; CHECK: lw      $1, {{[0-9]+}}($sp)
; CHECK: lw      $gp, {{[0-9]+}}($sp)
; CHECK: lw      $ra, [[R5:[0-9]+]]($sp)
; CHECK: lw      $8, {{[0-9]+}}($sp)
; CHECK: lw      $9, {{[0-9]+}}($sp)
; CHECK: lw      $10, {{[0-9]+}}($sp)
; CHECK: lw      $11, {{[0-9]+}}($sp)
; CHECK: lw      $12, {{[0-9]+}}($sp)
; CHECK: lw      $13, {{[0-9]+}}($sp)
; CHECK: lw      $14, {{[0-9]+}}($sp)
; CHECK: lw      $15, {{[0-9]+}}($sp)
; CHECK: lw      $24, {{[0-9]+}}($sp)
; CHECK: lw      $25, {{[0-9]+}}($sp)
; CHECK: lw      $2, {{[0-9]+}}($sp)
; CHECK: lw      $3, {{[0-9]+}}($sp)
; CHECK: lw      $4, {{[0-9]+}}($sp)
; CHECK: lw      $5, {{[0-9]+}}($sp)
; CHECK: lw      $6, {{[0-9]+}}($sp)
; CHECK: lw      $7, {{[0-9]+}}($sp)
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, [[R2:[0-9]+]]($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, [[R1:[0-9]+]]($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
  ret void
}

declare void @write(...)

define void @isr_sw1() #2 {
; CHECK-LABEL: isr_sw1:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 2
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

define void @isr_hw0() #3 {
; CHECK-LABEL: isr_hw0:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 3
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

define void @isr_hw1() #4 {
; CHECK-LABEL: isr_hw1:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 4
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }


define void @isr_hw2() #5 {
; CHECK-LABEL: isr_hw2:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 5
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

define void @isr_hw3() #6 {
; CHECK-LABEL: isr_hw3:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 6
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

define void @isr_hw4() #7 {
; CHECK-LABEL: isr_hw4:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 7
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

define void @isr_hw5() #8 {
; CHECK-LABEL: isr_hw5:
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $zero, 8, 8
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

define void @isr_eic() #9 {
; CHECK-LABEL: isr_eic:
; CHECK: mfc0   $26, $13, 0
; CHECK: ext    $26, $26, 10, 6
; CHECK: mfc0   $27, $14, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: mfc0   $27, $12, 0
; CHECK: sw     $27, {{[0-9]+}}($sp)
; CHECK: ins    $27, $26, 10, 6
; CHECK: ins    $27, $zero, 1, 4
; CHECK: ins    $27, $zero, 29, 1
; CHECK: mtc0   $27, $12, 0
  ret void
; CHECK: di
; CHECK: ehb
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $14, 0
; CHECK: lw      $27, {{[0-9]+}}($sp)
; CHECK: mtc0    $27, $12, 0
; CHECK: eret
 }

attributes #0 = { "interrupt"="sw0" }
attributes #2 = { "interrupt"="sw1" }
attributes #3 = { "interrupt"="hw0" }
attributes #4 = { "interrupt"="hw1" }
attributes #5 = { "interrupt"="hw2" }
attributes #6 = { "interrupt"="hw3" }
attributes #7 = { "interrupt"="hw4" }
attributes #8 = { "interrupt"="hw5" }
attributes #9 = { "interrupt"="eic" }
