; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP32,PRE4
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP32,GP32-CMOV
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32,GP32-CMOV
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32,GP32-CMOV
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32,GP32-CMOV
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP64,GP64-NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP64,GP64-NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP64,GP64-NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64,GP64-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64,GP64-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64,GP64-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64,GP64-R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips -O2 -verify-machineinstrs | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR3,MM32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -O2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM32
; RUN: llc < %s -march=mips -mcpu=mips64r6 -target-abi n64 -mattr=+micromips -O2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM64


; FIXME: This code sequence is inefficient as it should be 'subu $[[T0]], $zero, $[[T0]'. 
; This sequence is even better as it's a single instruction. See D25485 for the rest of 
; the cases where this sequence occurs.

define signext i1 @add_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: add_i1:

  ; NOT-R2-R6:  addu   $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:  andi   $[[T0]], $[[T0]], 1
  ; NOT-R2-R6:  negu   $2, $[[T0]]

  ; R2-R6:      addu   $[[T0:[0-9]+]], $4, $5
  ; R2-R6:      andi   $[[T0]], $[[T0]], 1
  ; R2-R6:      negu   $2, $[[T0]]

  ; MMR6:       addu16  $[[T0:[0-9]+]], $4, $5
  ; MMR6:       andi16  $[[T0]], $[[T0]], 1
  ; MMR6:       li16    $[[T1:[0-9]+]], 0
  ; MMR6:       subu16  $[[T0]], $[[T1]], $[[T0]]

  %r = add i1 %a, %b
  ret i1 %r
}

define signext i8 @add_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: add_i8:

  ; NOT-R2-R6:  addu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:  sll     $[[T0]], $[[T0]], 24
  ; NOT-R2-R6:  sra     $2, $[[T0]], 24

  ; R2-R6:      addu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:      seb     $2, $[[T0:[0-9]+]]

  ; MMR6:       addu16  $[[T0:[0-9]+]], $4, $5
  ; MMR6:       seb     $2, $[[T0]]

  %r = add i8 %a, %b
  ret i8 %r
}

define signext i16 @add_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: add_i16:

  ; NOT-R2-R6:  addu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:  sll     $[[T0]], $[[T0]], 16
  ; NOT-R2-R6:  sra     $2, $[[T0]], 16

  ; R2-R6:      addu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:      seh     $2, $[[T0]]

  ; MMR6:       addu16  $[[T0:[0-9]+]], $4, $5
  ; MMR6:       seh     $2, $[[T0]]

  %r = add i16 %a, %b
  ret i16 %r
}

define signext i32 @add_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: add_i32:

  ; NOT-R2-R6:  addu    $2, $4, $5
  ; R2-R6:      addu    $2, $4, $5

  ; MMR6:       addu16  $[[T0:[0-9]+]], $4, $5

  %r = add i32 %a, %b
  ret i32 %r
}

define signext i64 @add_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: add_i64:

  ; GP32-DAG:   addu    $[[T0:[0-9]+]], $4, $6
  ; GP32-DAG:   addu    $3, $5, $7
  ; GP32:       sltu    $[[T1:[0-9]+]], $3, $5
  ; GP32:       addu    $2, $[[T0]], $[[T1]]

  ; GP64:       daddu   $2, $4, $5

  ; MM32-DAG:   addu16  $3, $5, $7
  ; MM32-DAG:   addu16  $[[T0:[0-9]+]], $4, $6
  ; MM32:       sltu    $[[T1:[0-9]+]], $3, $5
  ; MM32:       addu16  $2, $[[T0]], $[[T1]]

  ; MM64:       daddu   $2, $4, $5

  %r = add i64 %a, %b
  ret i64 %r
}

define signext i128 @add_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: add_i128:

  ; PRE4:       move    $[[R1:[0-9]+]], $5
  ; PRE4:       move    $[[R2:[0-9]+]], $4
  ; PRE4:       lw   $[[R3:[0-9]+]], 24($sp)
  ; PRE4:       addu   $[[R4:[0-9]+]], $6, $[[R3]]
  ; PRE4:       lw   $[[R5:[0-9]+]], 28($sp)
  ; PRE4:       addu   $[[R6:[0-9]+]], $7, $[[R5]]
  ; PRE4:       sltu   $[[R7:[0-9]+]], $[[R6]], $7
  ; PRE4:       addu   $[[R8:[0-9]+]], $[[R4]], $[[R7]]
  ; PRE4:       xor   $[[R9:[0-9]+]], $[[R8]], $6
  ; PRE4:       sltiu   $[[R10:[0-9]+]], $[[R9]], 1
  ; PRE4:       bnez   $[[R10]], $BB5_2
  ; PRE4:       sltu   $[[R7]], $[[R8]], $6
  ; PRE4:       lw   $[[R12:[0-9]+]], 20($sp)
  ; PRE4:       addu   $[[R13:[0-9]+]], $[[R1]], $[[R12]]
  ; PRE4:       lw   $[[R14:[0-9]+]], 16($sp)
  ; PRE4:       addu   $[[R15:[0-9]+]], $[[R13]], $[[R7]]
  ; PRE4:       addu   $[[R16:[0-9]+]], $[[R2]], $[[R14]]
  ; PRE4:       sltu   $[[R17:[0-9]+]], $[[R15]], $[[R13]]
  ; PRE4:       sltu   $[[R18:[0-9]+]], $[[R13]], $[[R1]]
  ; PRE4:       addu   $[[R19:[0-9]+]], $[[R16]], $[[R18]]
  ; PRE4:       addu   $2, $[[R19]], $[[R17]]

  ; GP32-CMOV:  lw        $[[T0:[0-9]+]], 24($sp)
  ; GP32-CMOV:  addu      $[[T1:[0-9]+]], $6, $[[T0]]
  ; GP32-CMOV:  lw        $[[T2:[0-9]+]], 28($sp)
  ; GP32-CMOV:  addu      $[[T3:[0-9]+]], $7, $[[T2]]
  ; GP32-CMOV:  sltu      $[[T4:[0-9]+]], $[[T3]], $7
  ; GP32-CMOV:  addu      $[[T5:[0-9]+]], $[[T1]], $[[T4]]
  ; GP32-CMOV:  sltu      $[[T6:[0-9]+]], $[[T5]], $6
  ; GP32-CMOV:  xor       $[[T7:[0-9]+]], $[[T5]], $6
  ; GP32-CMOV:  movz      $[[T8:[0-9]+]], $[[T4]], $[[T7]]
  ; GP32-CMOV:  lw        $[[T9:[0-9]+]], 20($sp)
  ; GP32-CMOV:  addu      $[[T10:[0-9]+]], $5, $[[T4]]
  ; GP32-CMOV:  addu      $[[T11:[0-9]+]], $[[T10]], $[[T8]]
  ; GP32-CMOV:  lw        $[[T12:[0-9]+]], 16($sp)
  ; GP32-CMOV:  sltu      $[[T13:[0-9]+]], $[[T11]], $[[T10]]
  ; GP32-CMOV:  addu      $[[T14:[0-9]+]], $4, $[[T12]]
  ; GP32-CMOV:  sltu      $[[T15:[0-9]+]], $[[T10]], $5
  ; GP32-CMOV:  addu      $[[T16:[0-9]+]], $[[T14]], $[[T15]]
  ; GP32-CMOV:  addu      $[[T17:[0-9]+]], $[[T16]], $[[T13]]
  ; GP32-CMOV:  move      $4, $[[T5]]
  ; GP32-CMOV:  move      $5, $[[T3]]

  ; GP64:           daddu   $[[T0:[0-9]+]], $4, $6
  ; GP64:           daddu   $[[T1:[0-9]+]], $5, $7
  ; GP64:           sltu    $[[T2:[0-9]+]], $[[T1]], $5
  ; GP64-NOT-R2-R6: dsll    $[[T3:[0-9]+]], $[[T2]], 32
  ; GP64-NOT-R2-R6: dsrl    $[[T4:[0-9]+]], $[[T3]], 32
  ; GP64-R2-R6:     dext    $[[T4:[0-9]+]], $[[T2]], 0, 32

  ; GP64:           daddu   $2, $[[T0]], $[[T4]]

  ; MMR3:       move      $[[T1:[0-9]+]], $5
  ; MMR3-DAG:   lw        $[[T2:[0-9]+]], 32($sp)
  ; MMR3:       addu16    $[[T3:[0-9]+]], $6, $[[T2]]
  ; MMR3-DAG:   lw        $[[T4:[0-9]+]], 36($sp)
  ; MMR3:       addu16    $[[T5:[0-9]+]], $7, $[[T4]]
  ; MMR3:       sltu      $[[T6:[0-9]+]], $[[T5]], $7
  ; MMR3:       addu16    $[[T7:[0-9]+]], $[[T3]], $[[T6]]
  ; MMR3:       sltu      $[[T8:[0-9]+]], $[[T7]], $6
  ; MMR3:       xor       $[[T9:[0-9]+]], $[[T7]], $6
  ; MMR3:       movz      $[[T8]], $[[T6]], $[[T9]]
  ; MMR3:       lw        $[[T10:[0-9]+]], 28($sp)
  ; MMR3:       addu16    $[[T11:[0-9]+]], $[[T1]], $[[T10]]
  ; MMR3:       addu16    $[[T12:[0-9]+]], $[[T11]], $[[T8]]
  ; MMR3:       lw        $[[T13:[0-9]+]], 24($sp)
  ; MMR3:       sltu      $[[T14:[0-9]+]], $[[T12]], $[[T11]]
  ; MMR3:       addu16    $[[T15:[0-9]+]], $4, $[[T13]]
  ; MMR3:       sltu      $[[T16:[0-9]+]], $[[T11]], $[[T1]]
  ; MMR3:       addu16    $[[T17:[0-9]+]], $[[T15]], $[[T16]]
  ; MMR3:       addu16    $2, $2, $[[T14]]

  ; MMR6:        move      $[[T1:[0-9]+]], $5
  ; MMR6:        move      $[[T2:[0-9]+]], $4
  ; MMR6:        lw        $[[T3:[0-9]+]], 32($sp)
  ; MMR6:        addu16    $[[T4:[0-9]+]], $6, $[[T3]]
  ; MMR6:        lw        $[[T5:[0-9]+]], 36($sp)
  ; MMR6:        addu16    $[[T6:[0-9]+]], $7, $[[T5]]
  ; MMR6:        sltu      $[[T7:[0-9]+]], $[[T6]], $7
  ; MMR6:        addu16    $[[T8:[0-9]+]], $[[T4]], $7
  ; MMR6:        sltu      $[[T9:[0-9]+]], $[[T8]], $6
  ; MMR6:        xor       $[[T10:[0-9]+]], $[[T4]], $6
  ; MMR6:        sltiu     $[[T11:[0-9]+]], $[[T10]], 1
  ; MMR6:        seleqz    $[[T12:[0-9]+]], $[[T9]], $[[T11]]
  ; MMR6:        selnez    $[[T13:[0-9]+]], $[[T7]], $[[T11]]
  ; MMR6:        lw        $[[T14:[0-9]+]], 24($sp)
  ; MMR6:        or        $[[T15:[0-9]+]], $[[T13]], $[[T12]]
  ; MMR6:        addu16    $[[T16:[0-9]+]], $[[T2]], $[[T14]]
  ; MMR6:        lw        $[[T17:[0-9]+]], 28($sp)
  ; MMR6:        addu16    $[[T18:[0-9]+]], $[[T1]], $[[T17]]
  ; MMR6:        addu16    $[[T19:[0-9]+]], $[[T18]], $[[T15]]
  ; MMR6:        sltu      $[[T20:[0-9]+]], $[[T18]], $[[T1]]
  ; MMR6:        sltu      $[[T21:[0-9]+]], $[[T17]], $[[T18]]
  ; MMR6:        addu16    $2, $[[T16]], $[[T20]]
  ; MMR6:        addu16    $2, $[[T20]], $[[T21]]

  ; MM64:       daddu     $[[T0:[0-9]+]], $4, $6
  ; MM64:       daddu     $3, $5, $7
  ; MM64:       sltu      $[[T1:[0-9]+]], $3, $5
  ; MM64:       dsll      $[[T2:[0-9]+]], $[[T1]], 32
  ; MM64:       dsrl      $[[T3:[0-9]+]], $[[T2]], 32
  ; MM64:       daddu     $2, $[[T0]], $[[T3]]

  %r = add i128 %a, %b
  ret i128 %r
}

define signext i1 @add_i1_4(i1 signext %a) {
; ALL-LABEL: add_i1_4:

  ; ALL:        move      $2, $4

  %r = add i1 4, %a
  ret i1 %r
}

define signext i8 @add_i8_4(i8 signext %a) {
; ALL-LABEL: add_i8_4:

  ; NOT-R2-R6:  sll     $[[T0:[0-9]+]], $4, 24
  ; NOT-R2-R6:  lui     $[[T1:[0-9]+]], 1024
  ; NOT-R2-R6:  addu    $[[T0]], $[[T0]], $[[T1]]
  ; NOT-R2-R6:  sra     $2, $[[T0]], 24

  ; R2-R6:      addiu   $[[T0:[0-9]+]], $4, 4
  ; R2-R6:      seb     $2, $[[T0]]

  ; MM32:       addiur2 $[[T0:[0-9]+]], $4, 4
  ; MM32:       seb     $2, $[[T0]]

  ; MM64:       addiur2 $[[T0:[0-9]+]], $4, 4
  ; MM64:       seb     $2, $[[T0]]

  %r = add i8 4, %a
  ret i8 %r
}

define signext i16 @add_i16_4(i16 signext %a) {
; ALL-LABEL: add_i16_4:

  ; NOT-R2-R6:  sll     $[[T0:[0-9]+]], $4, 16
  ; NOT-R2-R6:  lui     $[[T1:[0-9]+]], 4
  ; NOT-R2-R6:  addu    $[[T0]], $[[T0]], $[[T1]]
  ; NOT-R2-R6:  sra     $2, $[[T0]], 16

  ; R2-R6:      addiu   $[[T0:[0-9]+]], $4, 4
  ; R2-R6:      seh     $2, $[[T0]]

  ; MM32:       addiur2 $[[T0:[0-9]+]], $4, 4
  ; MM32:       seh     $2, $[[T0]]

  ; MM64:       addiur2 $[[T0:[0-9]+]], $4, 4
  ; MM64:       seh     $2, $[[T0]]

  %r = add i16 4, %a
  ret i16 %r
}

define signext i32 @add_i32_4(i32 signext %a) {
; ALL-LABEL: add_i32_4:

  ; GP32:       addiu   $2, $4, 4

  ; GP64:       addiu   $2, $4, 4

  ; MM32:       addiur2 $2, $4, 4

  ; MM64:       addiur2 $2, $4, 4

  %r = add i32 4, %a
  ret i32 %r
}

define signext i64 @add_i64_4(i64 signext %a) {
; ALL-LABEL: add_i64_4:

  ; GP32:       addiu   $3, $5, 4
  ; GP32:       sltu    $[[T0:[0-9]+]], $3, $5
  ; GP32:       addu    $2, $4, $[[T0]]

  ; MM32:       addiur2 $[[T1:[0-9]+]], $5, 4
  ; MM32:       sltu    $[[T2:[0-9]+]], $[[T1]], $5
  ; MM32:       addu16  $2, $4, $[[T2]]

  ; GP64:       daddiu  $2, $4, 4


  ; MM64:       daddiu  $2, $4, 4

  %r = add i64 4, %a
  ret i64 %r
}

define signext i128 @add_i128_4(i128 signext %a) {
; ALL-LABEL: add_i128_4:

  ; PRE4:       move   $[[T0:[0-9]+]], $5
  ; PRE4:       addiu  $[[T1:[0-9]+]], $7, 4
  ; PRE4:       sltu   $[[T2:[0-9]+]], $[[T1]], $7
  ; PRE4:       xori   $[[T3:[0-9]+]], $[[T2]], 1
  ; PRE4:       bnez   $[[T3]], $BB[[BB0:[0-9_]+]]
  ; PRE4:       addu   $[[T4:[0-9]+]], $6, $[[T2]]
  ; PRE4:       sltu   $[[T5:[0-9]+]], $[[T4]], $6
  ; PRE4;       $BB[[BB0:[0-9]+]]:
  ; PRE4:       addu   $[[T6:[0-9]+]], $[[T0]], $[[T5]]
  ; PRE4:       sltu   $[[T7:[0-9]+]], $[[T6]], $[[T0]]
  ; PRE4:       addu   $[[T8:[0-9]+]], $4, $[[T7]]
  ; PRE4:       move    $4, $[[T4]]

  ; GP32-CMOV:  addiu   $[[T0:[0-9]+]], $7, 4
  ; GP32-CMOV:  sltu    $[[T1:[0-9]+]], $[[T0]], $7
  ; GP32-CMOV:  addu    $[[T2:[0-9]+]], $6, $[[T1]]
  ; GP32-CMOV:  sltu    $[[T3:[0-9]+]], $[[T2]], $6
  ; GP32-CMOV:  movz    $[[T3]], $[[T1]], $[[T1]]
  ; GP32-CMOV:  addu    $[[T4:[0-9]+]], $5, $[[T3]]
  ; GP32-CMOV:  sltu    $[[T5:[0-9]+]], $[[T4]], $5
  ; GP32-CMOV:  addu    $[[T7:[0-9]+]], $4, $[[T5]]
  ; GP32-CMOV:  move    $4, $[[T2]]
  ; GP32-CMOV:  move    $5, $[[T0]]

  ; GP64:           daddiu  $[[T0:[0-9]+]], $5, 4
  ; GP64:           sltu    $[[T1:[0-9]+]], $[[T0]], $5
  ; GP64-NOT-R2-R6: dsll    $[[T2:[0-9]+]], $[[T1]], 32
  ; GP64-NOT-R2-R6: dsrl    $[[T3:[0-9]+]], $[[T2]], 32
  ; GP64-R2-R6:     dext    $[[T3:[0-9]+]], $[[T1]], 0, 32

  ; GP64:           daddu   $2, $4, $[[T3]]

  ; MMR3:       addiur2 $[[T0:[0-9]+]], $7, 4
  ; MMR3:       sltu    $[[T1:[0-9]+]], $[[T0]], $7
  ; MMR3:       sltu    $[[T2:[0-9]+]], $[[T0]], $7
  ; MMR3:       addu16  $[[T3:[0-9]+]], $6, $[[T2]]
  ; MMR3:       sltu    $[[T4:[0-9]+]], $[[T3]], $6
  ; MMR3:       movz    $[[T4]], $[[T2]], $[[T1]]
  ; MMR3:       addu16  $[[T6:[0-9]+]], $5, $[[T4]]
  ; MMR3:       sltu    $[[T7:[0-9]+]], $[[T6]], $5
  ; MMR3:       addu16  $2, $4, $[[T7]]

  ; MMR6: addiur2 $[[T1:[0-9]+]], $7, 4
  ; MMR6: sltu    $[[T2:[0-9]+]], $[[T1]], $7
  ; MMR6: xori    $[[T3:[0-9]+]], $[[T2]], 1
  ; MMR6: selnez  $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; MMR6: addu16  $[[T5:[0-9]+]], $6, $[[T2]]
  ; MMR6: sltu    $[[T6:[0-9]+]], $[[T5]], $6
  ; MMR6: seleqz  $[[T7:[0-9]+]], $[[T6]], $[[T3]]
  ; MMR6: or      $[[T8:[0-9]+]], $[[T4]], $[[T7]]
  ; MMR6: addu16  $[[T9:[0-9]+]], $5, $[[T8]]
  ; MMR6: sltu    $[[T10:[0-9]+]], $[[T9]], $5
  ; MMR6: addu16  $[[T11:[0-9]+]], $4, $[[T10]]
  ; MMR6: move    $4, $7
  ; MMR6: move    $5, $[[T1]]

  ; MM64:       daddiu  $[[T0:[0-9]+]], $5, 4
  ; MM64:       sltu    $[[T1:[0-9]+]], $[[T0]], $5
  ; MM64:       dsll    $[[T2:[0-9]+]], $[[T1]], 32
  ; MM64:       dsrl    $[[T3:[0-9]+]], $[[T2]], 32
  ; MM64:       daddu   $2, $4, $[[T3]]

  %r = add i128 4, %a
  ret i128 %r
}

define signext i1 @add_i1_3(i1 signext %a) {
; ALL-LABEL: add_i1_3:
  ; GP32:        addiu  $[[T0:[0-9]+]], $4, 1
  ; GP32:        andi   $[[T0]], $[[T0]], 1
  ; GP32:        negu   $2, $[[T0]]

  ; GP64:        addiu  $[[T0:[0-9]+]], $4, 1
  ; GP64:        andi   $[[T0]], $[[T0]], 1
  ; GP64:        negu   $2, $[[T0]]

  ; MMR6:        addiur2 $[[T0:[0-9]+]], $4, 1
  ; MMR6:        andi16  $[[T0]], $[[T0]], 1
  ; MMR6:        li16    $[[T1:[0-9]+]], 0
  ; MMR6:        subu16  $2, $[[T1]], $[[T0]]

  %r = add i1 3, %a
  ret i1 %r
}

define signext i8 @add_i8_3(i8 signext %a) {
; ALL-LABEL: add_i8_3:

  ; NOT-R2-R6:  sll     $[[T0:[0-9]+]], $4, 24
  ; NOT-R2-R6:  lui     $[[T1:[0-9]+]], 768
  ; NOT-R2-R6:  addu    $[[T0]], $[[T0]], $[[T1]]
  ; NOT-R2-R6:  sra     $2, $[[T0]], 24

  ; R2-R6:      addiu   $[[T0:[0-9]+]], $4, 3
  ; R2-R6:      seb     $2, $[[T0]]

  ; MMR6:       addius5 $[[T0:[0-9]+]], 3
  ; MMR6:       seb     $2, $[[T0]]

  %r = add i8 3, %a
  ret i8 %r
}

define signext i16 @add_i16_3(i16 signext %a) {
; ALL-LABEL: add_i16_3:

  ; NOT-R2-R6:  sll     $[[T0:[0-9]+]], $4, 16
  ; NOT-R2-R6:  lui     $[[T1:[0-9]+]], 3
  ; NOT-R2-R6:  addu    $[[T0]], $[[T0]], $[[T1]]
  ; NOT-R2-R6:  sra     $2, $[[T0]], 16

  ; R2-R6:      addiu   $[[T0:[0-9]+]], $4, 3
  ; R2-R6:      seh     $2, $[[T0]]

  ; MMR6:       addius5 $[[T0:[0-9]+]], 3
  ; MMR6:       seh     $2, $[[T0]]

  %r = add i16 3, %a
  ret i16 %r
}

define signext i32 @add_i32_3(i32 signext %a) {
; ALL-LABEL: add_i32_3:

  ; NOT-R2-R6:  addiu   $2, $4, 3

  ; R2-R6:      addiu   $2, $4, 3

  ; MMR6:       addius5 $[[T0:[0-9]+]], 3
  ; MMR6:       move    $2, $[[T0]]

  %r = add i32 3, %a
  ret i32 %r
}

define signext i64 @add_i64_3(i64 signext %a) {
; ALL-LABEL: add_i64_3:

  ; GP32:       addiu   $[[T0:[0-9]+]], $5, 3
  ; GP32:       sltu    $[[T1:[0-9]+]], $[[T0]], $5
  ; GP32:       addu    $2, $4, $[[T1]]

  ; GP64:       daddiu  $2, $4, 3

  ; MM32:       move    $[[T1:[0-9]+]], $5
  ; MM32:       addius5 $[[T1]], 3
  ; MM32:       sltu    $[[T2:[0-9]+]], $[[T1]], $5
  ; MM32:       addu16  $2, $4, $[[T2]]

  ; MM64:       daddiu  $2, $4, 3

  %r = add i64 3, %a
  ret i64 %r
}

define signext i128 @add_i128_3(i128 signext %a) {
; ALL-LABEL: add_i128_3:

  ; PRE4:       move   $[[T0:[0-9]+]], $5
  ; PRE4:       addiu  $[[T1:[0-9]+]], $7, 3
  ; PRE4:       sltu   $[[T2:[0-9]+]], $[[T1]], $7
  ; PRE4:       xori   $[[T3:[0-9]+]], $[[T2]], 1
  ; PRE4:       bnez   $[[T3]], $BB[[BB0:[0-9_]+]]
  ; PRE4:       addu   $[[T4:[0-9]+]], $6, $[[T2]]
  ; PRE4:       sltu   $[[T5:[0-9]+]], $[[T4]], $6
  ; PRE4;       $BB[[BB0:[0-9]+]]:
  ; PRE4:       addu   $[[T6:[0-9]+]], $[[T0]], $[[T5]]
  ; PRE4:       sltu   $[[T7:[0-9]+]], $[[T6]], $[[T0]]
  ; PRE4:       addu   $[[T8:[0-9]+]], $4, $[[T7]]
  ; PRE4:       move    $4, $[[T4]]

  ; GP32-CMOV:  addiu   $[[T0:[0-9]+]], $7, 3
  ; GP32-CMOV:  sltu    $[[T1:[0-9]+]], $[[T0]], $7
  ; GP32-CMOV:  addu    $[[T2:[0-9]+]], $6, $[[T1]]
  ; GP32-CMOV:  sltu    $[[T3:[0-9]+]], $[[T2]], $6
  ; GP32-CMOV:  movz    $[[T3]], $[[T1]], $[[T1]]
  ; GP32-CMOV:  addu    $[[T4:[0-9]+]], $5, $[[T3]]
  ; GP32-CMOV:  sltu    $[[T5:[0-9]+]], $[[T4]], $5
  ; GP32-CMOV:  addu    $[[T7:[0-9]+]], $4, $[[T5]]
  ; GP32-CMOV:  move    $4, $[[T2]]
  ; GP32-CMOV:  move    $5, $[[T0]]

  ; GP64:           daddiu  $[[T0:[0-9]+]], $5, 3
  ; GP64:           sltu    $[[T1:[0-9]+]], $[[T0]], $5

  ; GP64-NOT-R2-R6: dsll    $[[T2:[0-9]+]], $[[T1]], 32
  ; GP64-NOT-R2-R6: dsrl    $[[T3:[0-9]+]], $[[T2]], 32
  ; GP64-R2-R6:     dext    $[[T3:[0-9]+]], $[[T1]], 0, 32

  ; GP64:           daddu   $2, $4, $[[T3]]

  ; MMR3:       move    $[[T1:[0-9]+]], $7
  ; MMR3:       addius5 $[[T1]], 3
  ; MMR3:       sltu    $[[T2:[0-9]+]], $[[T1]], $7
  ; MMR3:       sltu    $[[T3:[0-9]+]], $[[T1]], $7
  ; MMR3:       addu16  $[[T4:[0-9]+]], $6, $[[T3]]
  ; MMR3:       sltu    $[[T5:[0-9]+]], $[[T4]], $6
  ; MMR3:       movz    $[[T5]], $[[T3]], $[[T2]]
  ; MMR3:       addu16  $[[T6:[0-9]+]], $5, $[[T5]]
  ; MMR3:       sltu    $[[T7:[0-9]+]], $[[T6]], $5
  ; MMR3:       addu16  $2, $4, $[[T7]]

  ; MMR6: move    $[[T1:[0-9]+]], $7
  ; MMR6: addius5 $[[T1]], 3
  ; MMR6: sltu    $[[T2:[0-9]+]], $[[T1]], $7
  ; MMR6: xori    $[[T3:[0-9]+]], $[[T2]], 1
  ; MMR6: selnez  $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; MMR6: addu16  $[[T5:[0-9]+]], $6, $[[T2]]
  ; MMR6: sltu    $[[T6:[0-9]+]], $[[T5]], $6
  ; MMR6: seleqz  $[[T7:[0-9]+]], $[[T6]], $[[T3]]
  ; MMR6: or      $[[T8:[0-9]+]], $[[T4]], $[[T7]]
  ; MMR6: addu16  $[[T9:[0-9]+]], $5, $[[T8]]
  ; MMR6: sltu    $[[T10:[0-9]+]], $[[T9]], $5
  ; MMR6: addu16  $[[T11:[0-9]+]], $4, $[[T10]]
  ; MMR6: move    $4, $[[T5]]
  ; MMR6: move    $5, $[[T1]]

  ; MM64:       daddiu  $[[T0:[0-9]+]], $5, 3
  ; MM64:       sltu    $[[T1:[0-9]+]], $[[T0]], $5
  ; MM64:       dsll    $[[T2:[0-9]+]], $[[T1]], 32
  ; MM64:       dsrl    $[[T3:[0-9]+]], $[[T2]], 32
  ; MM64:       daddu   $2, $4, $[[T3]]

  %r = add i128 3, %a
  ret i128 %r
}
