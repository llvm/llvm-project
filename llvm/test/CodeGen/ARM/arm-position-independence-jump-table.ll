; Test for generation of jump table for ropi/rwpi

; RUN: llc -relocation-model=static    -mtriple=armv7a--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM --check-prefix=ARM_ABS
; RUN: llc -relocation-model=ropi      -mtriple=armv7a--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM --check-prefix=ARM_PC
; RUN: llc -relocation-model=ropi-rwpi -mtriple=armv7a--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM --check-prefix=ARM_PC

; RUN: llc -relocation-model=static    -mtriple=thumbv7m--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2
; RUN: llc -relocation-model=ropi      -mtriple=thumbv7m--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv7m--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2

; RUN: llc -relocation-model=static    -mtriple=thumbv6m--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1 --check-prefix=THUMB1_ABS
; RUN: llc -relocation-model=ropi      -mtriple=thumbv6m--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1 --check-prefix=THUMB1_PC
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv6m--none-eabi -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1 --check-prefix=THUMB1_PC


declare void @exit0()
declare void @exit1()
declare void @exit2()
declare void @exit3()
declare void @exit4()
define void @jump_table(i32 %val) {
entry:
  switch i32 %val, label %default [ i32 1, label %lab1
                                    i32 2, label %lab2
                                    i32 3, label %lab3
                                    i32 4, label %lab4 ]

default:
  tail call void @exit0()
  ret void

lab1:
  tail call void @exit1()
  ret void

lab2:
  tail call void @exit2()
  ret void

lab3:
  tail call void @exit3()
  ret void

lab4:
  tail call void @exit4()
  ret void

; CHECK-LABEL: jump_table:

; ARM: lsl     r[[R_TAB_IDX:[0-9]+]], r{{[0-9]+}}, #2
; ARM: adr     r[[R_TAB_BASE:[0-9]+]], [[LJTI:\.LJTI[0-9]+_[0-9]+]]
; ARM_ABS: ldr     pc, [r[[R_TAB_IDX]], r[[R_TAB_BASE]]]
; ARM_PC:  ldr     r[[R_OFFSET:[0-9]+]], [r[[R_TAB_IDX]], r[[R_TAB_BASE]]]
; ARM_PC:  add     pc, r[[R_OFFSET]], r[[R_TAB_BASE]]
; ARM: [[LJTI]]
; ARM_ABS: .long [[LBB1:\.LBB[0-9]+_[0-9]+]]
; ARM_ABS: .long [[LBB2:\.LBB[0-9]+_[0-9]+]]
; ARM_ABS: .long [[LBB3:\.LBB[0-9]+_[0-9]+]]
; ARM_ABS: .long [[LBB4:\.LBB[0-9]+_[0-9]+]]
; ARM_PC:  .long [[LBB1:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; ARM_PC:  .long [[LBB2:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; ARM_PC:  .long [[LBB3:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; ARM_PC:  .long [[LBB4:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; ARM: [[LBB1]]
; ARM-NEXT: b exit1
; ARM: [[LBB2]]
; ARM-NEXT: b exit2
; ARM: [[LBB3]]
; ARM-NEXT: b exit3
; ARM: [[LBB4]]
; ARM-NEXT: b exit4

; THUMB2: [[LCPI:\.LCPI[0-9]+_[0-9]+]]:
; THUMB2: tbb     [pc, r{{[0-9]+}}]
; THUMB2: .byte   ([[LBB1:\.LBB[0-9]+_[0-9]+]]-([[LCPI]]+4))/2
; THUMB2: .byte   ([[LBB2:\.LBB[0-9]+_[0-9]+]]-([[LCPI]]+4))/2
; THUMB2: .byte   ([[LBB3:\.LBB[0-9]+_[0-9]+]]-([[LCPI]]+4))/2
; THUMB2: .byte   ([[LBB4:\.LBB[0-9]+_[0-9]+]]-([[LCPI]]+4))/2
; THUMB2: [[LBB1]]
; THUMB2-NEXT: b exit1
; THUMB2: [[LBB2]]
; THUMB2-NEXT: b exit2
; THUMB2: [[LBB3]]
; THUMB2-NEXT: b exit3
; THUMB2: [[LBB4]]
; THUMB2-NEXT: b exit4

; THUMB1: lsls    r[[R_TAB_INDEX:[0-9]+]], r{{[0-9]+}}, #2 
; THUMB1: adr     r[[R_TAB_BASE:[0-9]+]], [[LJTI:\.LJTI[0-9]+_[0-9]+]]
; THUMB1: ldr     r[[R_BB_ADDR:[0-9]+]], [r[[R_TAB_INDEX]], r[[R_TAB_BASE]]]
; THUMB1_PC: adds    r[[R_BB_ADDR]], r[[R_BB_ADDR]], r[[R_TAB_BASE]]
; THUMB1: mov     pc, r[[R_BB_ADDR]]
; THUMB1: [[LJTI]]
; THUMB1_ABS: .long [[LBB1:\.LBB[0-9]+_[0-9]+]]+1
; THUMB1_ABS: .long [[LBB2:\.LBB[0-9]+_[0-9]+]]+1
; THUMB1_ABS: .long [[LBB3:\.LBB[0-9]+_[0-9]+]]+1
; THUMB1_ABS: .long [[LBB4:\.LBB[0-9]+_[0-9]+]]+1
; THUMB1_PC:  .long [[LBB1:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; THUMB1_PC:  .long [[LBB2:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; THUMB1_PC:  .long [[LBB3:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; THUMB1_PC:  .long [[LBB4:\.LBB[0-9]+_[0-9]+]]-[[LJTI]]
; THUMB1: [[LBB1]]
; THUMB1-NEXT: bl exit1
; THUMB1-NEXT: pop
; THUMB1: [[LBB2]]
; THUMB1-NEXT: bl exit2
; THUMB1-NEXT: pop
; THUMB1: [[LBB3]]
; THUMB1-NEXT: bl exit3
; THUMB1-NEXT: pop
; THUMB1: [[LBB4]]
; THUMB1-NEXT: bl exit4
; THUMB1-NEXT: pop
}
