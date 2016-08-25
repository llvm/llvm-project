; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,M2,M2-M3
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R1
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,SEL,SEL-32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,M3,M2-M3
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,SEL,SEL-64
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM32R3
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM32R6

define signext i1 @tst_select_i1_i1(i1 signext %s,
                                    i1 signext %x, i1 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i1:

  ; M2-M3:  andi    $[[T0:[0-9]+]], $4, 1
  ; M2:     bnez    $[[T0]], [[BB0:\$BB[0-9_]+]]
  ; M3:     bnez    $[[T0]], [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:  nop
  ; M2-M3:  move    $5, $6
  ; M2-M3:  [[BB0]]:
  ; M2-M3:  jr      $ra
  ; M2-M3:  move    $2, $5

  ; CMOV:   andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV:   movn    $6, $5, $[[T0]]
  ; CMOV:   move    $2, $6

  ; SEL:    andi    $[[T0:[0-9]+]], $4, 1
  ; SEL:    seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL:    selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; SEL:    or      $2, $[[T2]], $[[T1]]

  ; MM32R3:   andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R3:   movn    $[[T1:[0-9]+]], $5, $[[T0]]
  ; MM32R3:   move    $2, $[[T1]]

  ; MMR6:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MMR6:     seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; MMR6:     selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; MMR6:     or      $2, $[[T2]], $[[T1]]

  %r = select i1 %s, i1 %x, i1 %y
  ret i1 %r
}

define signext i8 @tst_select_i1_i8(i1 signext %s,
                                    i8 signext %x, i8 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i8:

  ; M2-M3:  andi    $[[T0:[0-9]+]], $4, 1
  ; M2:     bnez    $[[T0]], [[BB0:\$BB[0-9_]+]]
  ; M3:     bnez    $[[T0]], [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:  nop
  ; M2-M3:  move    $5, $6
  ; M2-M3:  [[BB0]]:
  ; M2-M3:  jr      $ra
  ; M2-M3:  move    $2, $5

  ; CMOV:   andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV:   movn    $6, $5, $[[T0]]
  ; CMOV:   move    $2, $6

  ; SEL:    andi    $[[T0:[0-9]+]], $4, 1
  ; SEL:    seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL:    selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; SEL:    or      $2, $[[T2]], $[[T1]]

  ; MM32R3:   andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R3:   movn    $[[T1:[0-9]+]], $5, $[[T0]]
  ; MM32R3:   move    $2, $[[T1]]

  ; MMR6:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MMR6:     seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; MMR6:     selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; MMR6:     or      $2, $[[T2]], $[[T1]]

  %r = select i1 %s, i8 %x, i8 %y
  ret i8 %r
}

define signext i32 @tst_select_i1_i32(i1 signext %s,
                                      i32 signext %x, i32 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i32:

  ; M2-M3:  andi    $[[T0:[0-9]+]], $4, 1
  ; M2:     bnez    $[[T0]], [[BB0:\$BB[0-9_]+]]
  ; M3:     bnez    $[[T0]], [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:  nop
  ; M2-M3:  move    $5, $6
  ; M2-M3:  [[BB0]]:
  ; M2-M3:  jr      $ra
  ; M2-M3:  move    $2, $5

  ; CMOV:   andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV:   movn    $6, $5, $[[T0]]
  ; CMOV:   move    $2, $6

  ; SEL:    andi    $[[T0:[0-9]+]], $4, 1
  ; SEL:    seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL:    selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; SEL:    or      $2, $[[T2]], $[[T1]]

  ; MM32R3:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R3:     movn    $[[T1:[0-9]+]], $5, $[[T0]]
  ; MM32R3:     move    $2, $[[T1]]

  ; MMR6:       andi16  $[[T0:[0-9]+]], $4, 1
  ; MMR6:       seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; MMR6:       selnez  $[[T2:[0-9]+]], $5, $[[T0]]
  ; MMR6:       or      $2, $[[T2]], $[[T1]]

  %r = select i1 %s, i32 %x, i32 %y
  ret i32 %r
}

define signext i64 @tst_select_i1_i64(i1 signext %s,
                                      i64 signext %x, i64 signext %y) {
entry:
  ; ALL-LABEL: tst_select_i1_i64:

  ; M2:     andi    $[[T0:[0-9]+]], $4, 1
  ; M2:     bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2:     nop
  ; M2:     lw      $[[T1:[0-9]+]], 16($sp)
  ; M2:     $[[BB0]]:
  ; FIXME: This branch is redundant
  ; M2:     bnez    $[[T0]], $[[BB1:BB[0-9_]+]]
  ; M2:     nop
  ; M2:     lw      $[[T2:[0-9]+]], 20($sp)
  ; M2:     $[[BB1]]:
  ; M2:     move    $2, $[[T1]]
  ; M2:     jr      $ra
  ; M2:     move    $3, $[[T2]]

  ; CMOV-32:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-32:    lw      $2, 16($sp)
  ; CMOV-32:    movn    $2, $6, $[[T0]]
  ; CMOV-32:    lw      $3, 20($sp)
  ; CMOV-32:    movn    $3, $7, $[[T0]]

  ; SEL-32:     andi    $[[T0:[0-9]+]], $4, 1
  ; SEL-32:     selnez  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL-32:     lw      $[[T2:[0-9]+]], 16($sp)
  ; SEL-32:     seleqz  $[[T3:[0-9]+]], $[[T2]], $[[T0]]
  ; SEL-32:     or      $2, $[[T1]], $[[T3]]
  ; SEL-32:     selnez  $[[T4:[0-9]+]], $7, $[[T0]]
  ; SEL-32:     lw      $[[T5:[0-9]+]], 20($sp)
  ; SEL-32:     seleqz  $[[T6:[0-9]+]], $[[T5]], $[[T0]]
  ; SEL-32:     or      $3, $[[T4]], $[[T6]]

  ; M3:         andi    $[[T0:[0-9]+]], $4, 1
  ; M3:         bnez    $[[T0]], [[BB0:\.LBB[0-9_]+]]
  ; M3:         nop
  ; M3:         move    $5, $6
  ; M3:         [[BB0]]:
  ; M3:         jr      $ra
  ; M3:         move    $2, $5

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-64:    movn    $6, $5, $[[T0]]
  ; CMOV-64:    move    $2, $6

  ; SEL-64:     andi    $[[T0:[0-9]+]], $4, 1
  ; FIXME: This shift is redundant
  ; SEL-64:     sll     $[[T0]], $[[T0]], 0
  ; SEL-64:     seleqz  $[[T1:[0-9]+]], $6, $[[T0]]
  ; SEL-64:     selnez  $[[T0]], $5, $[[T0]]
  ; SEL-64:     or      $2, $[[T0]], $[[T1]]

  ; MM32R3:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R3:     lw      $2, 16($sp)
  ; MM32R3:     movn    $2, $6, $[[T0]]
  ; MM32R3:     lw      $3, 20($sp)
  ; MM32R3:     movn    $3, $7, $[[T0]]

  ; MM32R6:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R6:     selnez  $[[T1:[0-9]+]], $6, $[[T0]]
  ; MM32R6:     lw      $[[T2:[0-9]+]], 16($sp)
  ; MM32R6:     seleqz  $[[T3:[0-9]+]], $[[T2]], $[[T0]]
  ; MM32R6:     or      $2, $[[T1]], $[[T3]]
  ; MM32R6:     selnez  $[[T6:[0-9]+]], $7, $[[T0]]
  ; MM32R6:     lw      $[[T4:[0-9]+]], 20($sp)
  ; MM32R6:     seleqz  $[[T5:[0-9]+]], $[[T4]], $[[T0]]
  ; MM32R6:     or      $3, $[[T6]], $[[T5]]

  %r = select i1 %s, i64 %x, i64 %y
  ret i64 %r
}

define i8* @tst_select_word_cst(i8* %a, i8* %b) {
  ; ALL-LABEL: tst_select_word_cst:

  ; M2:         addiu   $[[T0:[0-9]+]], $zero, -1
  ; M2:         xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; M2:         sltu    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; M2:         bnez    $[[T2]], [[BB0:\$BB[0-9_]+]]
  ; M2:         addiu   $2, $zero, 0
  ; M2:         move    $2, $4
  ; M2: [[BB0]]:
  ; M2:         jr      $ra

  ; M3:         daddiu  $[[T0:[0-9]+]], $zero, -1
  ; M3:         xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; M3:         sltu    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; M3:         bnez    $[[T2]], [[BB0:\.LBB[0-9_]+]]
  ; M3:         daddiu  $2, $zero, 0
  ; M3:         move    $2, $4
  ; M3: [[BB0]]:
  ; M3:         jr      $ra

  ; CMOV-32:    addiu   $[[T0:[0-9]+]], $zero, -1
  ; CMOV-32:    xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; CMOV-32:    movn    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; CMOV-32:    jr      $ra
  ; CMOV-32:    move    $2, $[[T2]]

  ; SEL-32:     addiu   $[[T0:[0-9]+]], $zero, -1
  ; SEL-32:     xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; SEL-32:     sltu    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; SEL-32:     jr      $ra
  ; SEL-32:     seleqz  $2, $4, $[[T2]]

  ; CMOV-64:    daddiu  $[[T0:[0-9]+]], $zero, -1
  ; CMOV-64:    xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; CMOV-64:    movn    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; CMOV-64:    move    $2, $[[T2]]

  ; SEL-64:     daddiu  $[[T0:[0-9]+]], $zero, -1
  ; SEL-64:     xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; SEL-64:     sltu    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; FIXME: This shift is redundant.
  ; SEL-64:     sll     $[[T2]], $[[T2]], 0
  ; SEL-64:     seleqz  $2, $4, $[[T2]]

  ; MM32R3:     li16    $[[T0:[0-9]+]], -1
  ; MM32R3:     xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; MM32R3:     li16    $[[T2:[0-9]+]], 0
  ; MM32R3:     movn    $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; MM32R3:     move    $2, $[[T3]]

  ; MM32R6:     li16    $[[T0:[0-9]+]], -1
  ; MM32R6:     xor     $[[T1:[0-9]+]], $5, $[[T0]]
  ; MM32R6:     sltu    $[[T2:[0-9]+]], $zero, $[[T1]]
  ; MM32R6:     seleqz  $2, $4, $[[T2]]

  %cmp = icmp eq i8* %b, inttoptr (i64 -1 to i8*)
  %r = select i1 %cmp, i8* %a, i8* null
  ret i8* %r
}
