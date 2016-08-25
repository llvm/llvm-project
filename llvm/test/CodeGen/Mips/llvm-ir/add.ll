; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R2-R6,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,R2-R6,GP64
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips -O2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -O2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM32
; RUN: llc < %s -march=mips -mcpu=mips64r6 -target-abi n64 -mattr=+micromips -O2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM64

define signext i1 @add_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: add_i1:

  ; NOT-R2-R6:  addu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:  sll     $[[T0]], $[[T0]], 31
  ; NOT-R2-R6:  sra     $2, $[[T0]], 31

  ; R2-R6:      addu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:      sll     $[[T0]], $[[T0]], 31
  ; R2-R6:      sra     $2, $[[T0]], 31

  ; MMR6:       addu16  $[[T0:[0-9]+]], $4, $5
  ; MMR6:       sll     $[[T1:[0-9]+]], $[[T0]], 31
  ; MMR6:       sra     $2, $[[T1]], 31

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

  ; GP32:       addu    $3, $5, $7
  ; GP32:       sltu    $[[T0:[0-9]+]], $3, $7
  ; GP32:       addu    $[[T1:[0-9]+]], $[[T0]], $6
  ; GP32:       addu    $2, $4, $[[T1]]

  ; GP64:       daddu   $2, $4, $5

  ; MM32:       addu    $3, $5, $7
  ; MM32:       sltu    $[[T0:[0-9]+]], $3, $7
  ; MM32:       addu    $[[T1:[0-9]+]], $[[T0]], $6
  ; MM32:       addu    $2, $4, $[[T1]]

  ; MM64:       daddu   $2, $4, $5

  %r = add i64 %a, %b
  ret i64 %r
}

define signext i128 @add_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: add_i128:

  ; GP32:       lw        $[[T0:[0-9]+]], 28($sp)
  ; GP32:       addu      $[[T1:[0-9]+]], $7, $[[T0]]
  ; GP32:       sltu      $[[T2:[0-9]+]], $[[T1]], $[[T0]]
  ; GP32:       lw        $[[T3:[0-9]+]], 24($sp)
  ; GP32:       addu      $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; GP32:       addu      $[[T5:[0-9]+]], $6, $[[T4]]
  ; GP32:       lw        $[[T6:[0-9]+]], 16($sp)
  ; GP32:       lw        $[[T7:[0-9]+]], 20($sp)
  ; GP32:       sltu      $[[T8:[0-9]+]], $[[T5]], $[[T3]]
  ; GP32:       addu      $[[T9:[0-9]+]], $[[T8]], $[[T7]]
  ; GP32:       addu      $3, $5, $[[T8]]
  ; GP32:       sltu      $[[T10:[0-9]+]], $3, $[[T7]]
  ; GP32:       addu      $[[T11:[0-9]+]], $[[T10]], $[[T6]]
  ; GP32:       addu      $2, $4, $[[T11]]
  ; GP32:       move      $4, $[[T5]]
  ; GP32:       move      $5, $[[T1]]

  ; GP64:       daddu     $3, $5, $7
  ; GP64:       sltu      $[[T0:[0-9]+]], $3, $7
  ; GP64:       daddu     $[[T1:[0-9]+]], $[[T0]], $6
  ; GP64:       daddu     $2, $4, $[[T1]]

  ; MM32:       lw        $[[T0:[0-9]+]], 28($sp)
  ; MM32:       addu      $[[T1:[0-9]+]], $7, $[[T0]]
  ; MM32:       sltu      $[[T2:[0-9]+]], $[[T1]], $[[T0]]
  ; MM32:       lw        $[[T3:[0-9]+]], 24($sp)
  ; MM32:       addu      $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; MM32:       addu      $[[T5:[0-9]+]], $6, $[[T4]]
  ; MM32:       lw        $[[T6:[0-9]+]], 16($sp)
  ; MM32:       lw        $[[T7:[0-9]+]], 20($sp)
  ; MM32:       sltu      $[[T8:[0-9]+]], $[[T5]], $[[T3]]
  ; MM32:       addu      $[[T9:[0-9]+]], $[[T8]], $[[T7]]
  ; MM32:       addu      $[[T10:[0-9]+]], $5, $[[T9]]
  ; MM32:       sltu      $[[T11:[0-6]+]], $[[T9]], $[[T7]]
  ; MM32:       addu      $[[T12:[0-6]+]], $[[T11]], $[[T6]]
  ; MM32:       addu      $[[T13:[0-9]+]], $4, $[[T12]]
  ; MM32:       move      $4, $[[T5]]
  ; MM32:       move      $5, $[[T1]]

  ; MM64:       daddu     $3, $5, $7
  ; MM64:       sltu      $[[T0:[0-9]+]], $3, $7
  ; MM64:       daddu     $[[T1:[0-9]+]], $[[T0]], $6
  ; MM64:       daddu     $2, $4, $[[T1]]

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

  ; GP32:       addiu   $[[T0:[0-9]+]], $5, 4
  ; GP32:       addiu   $[[T1:[0-9]+]], $zero, 4
  ; GP32:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; GP32:       addu    $2, $4, $[[T1]]

  ; GP64:       daddiu  $2, $4, 4

  ; MM32:       addiu   $[[T0:[0-9]+]], $5, 4
  ; MM32:       li16    $[[T1:[0-9]+]], 4
  ; MM32:       sltu    $[[T2:[0-9]+]], $[[T0]], $[[T1]]
  ; MM32:       addu    $2, $4, $[[T2]]

  ; MM64:       daddiu  $2, $4, 4

  %r = add i64 4, %a
  ret i64 %r
}

define signext i128 @add_i128_4(i128 signext %a) {
; ALL-LABEL: add_i128_4:

  ; GP32:       addiu   $[[T0:[0-9]+]], $7, 4
  ; GP32:       addiu   $[[T1:[0-9]+]], $zero, 4
  ; GP32:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; GP32:       addu    $[[T2:[0-9]+]], $6, $[[T1]]
  ; GP32:       sltu    $[[T1]], $[[T2]], $zero
  ; GP32:       addu    $[[T3:[0-9]+]], $5, $[[T1]]
  ; GP32:       sltu    $[[T1]], $[[T3]], $zero
  ; GP32:       addu    $[[T1]], $4, $[[T1]]
  ; GP32:       move    $4, $[[T2]]
  ; GP32:       move    $5, $[[T0]]

  ; GP64:       daddiu  $[[T0:[0-9]+]], $5, 4
  ; GP64:       daddiu  $[[T1:[0-9]+]], $zero, 4
  ; GP64:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; GP64:       daddu   $2, $4, $[[T1]]

  ; MM32:       addiu   $[[T0:[0-9]+]], $7, 4
  ; MM32:       li16    $[[T1:[0-9]+]], 4
  ; MM32:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; MM32:       addu    $[[T2:[0-9]+]], $6, $[[T1]]
  ; MM32:       li16    $[[T1]], 0
  ; MM32:       sltu    $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; MM32:       addu    $[[T3]], $5, $[[T3]]
  ; MM32:       sltu    $[[T1]], $[[T3]], $[[T1]]
  ; MM32:       addu    $[[T1]], $4, $[[T1]]
  ; MM32:       move    $4, $[[T2]]
  ; MM32:       move    $5, $[[T0]]

  ; MM64:       daddiu  $[[T0:[0-9]+]], $5, 4
  ; MM64:       daddiu  $[[T1:[0-9]+]], $zero, 4
  ; MM64:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; MM64:       daddu   $2, $4, $[[T1]]

  %r = add i128 4, %a
  ret i128 %r
}

define signext i1 @add_i1_3(i1 signext %a) {
; ALL-LABEL: add_i1_3:

  ; ALL:        sll     $[[T0:[0-9]+]], $4, 31
  ; ALL:        lui     $[[T1:[0-9]+]], 32768

  ; GP32:       addu    $[[T0]], $[[T0]], $[[T1]]
  ; GP32:       sra     $[[T1]], $[[T0]], 31

  ; GP64:       addu    $[[T0]], $[[T0]], $[[T1]]
  ; GP64:       sra     $[[T1]], $[[T0]], 31

  ; MMR6:       addu16  $[[T0]], $[[T0]], $[[T1]]
  ; MMR6:       sra     $[[T0]], $[[T0]], 31

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
  ; GP32:       addiu   $[[T1:[0-9]+]], $zero, 3
  ; GP32:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; GP32:       addu    $2, $4, $[[T1]]

  ; GP64:       daddiu  $2, $4, 3

  ; MM32:       addiu   $[[T0:[0-9]+]], $5, 3
  ; MM32:       li16    $[[T1:[0-9]+]], 3
  ; MM32:       sltu    $[[T2:[0-9]+]], $[[T0]], $[[T1]]
  ; MM32:       addu    $2, $4, $[[T2]]

  ; MM64:       daddiu  $2, $4, 3

  %r = add i64 3, %a
  ret i64 %r
}

define signext i128 @add_i128_3(i128 signext %a) {
; ALL-LABEL: add_i128_3:

  ; GP32:       addiu   $[[T0:[0-9]+]], $7, 3
  ; GP32:       addiu   $[[T1:[0-9]+]], $zero, 3
  ; GP32:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; GP32:       addu    $[[T2:[0-9]+]], $6, $[[T1]]
  ; GP32:       sltu    $[[T3:[0-9]+]], $[[T2]], $zero
  ; GP32:       addu    $[[T4:[0-9]+]], $5, $[[T3]]
  ; GP32:       sltu    $[[T5:[0-9]+]], $[[T4]], $zero
  ; GP32:       addu    $[[T5]], $4, $[[T5]]
  ; GP32:       move    $4, $[[T2]]
  ; GP32:       move    $5, $[[T0]]

  ; GP64:       daddiu  $[[T0:[0-9]+]], $5, 3
  ; GP64:       daddiu  $[[T1:[0-9]+]], $zero, 3
  ; GP64:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; GP64:       daddu   $2, $4, $[[T1]]

  ; MM32:       addiu   $[[T0:[0-9]+]], $7, 3
  ; MM32:       li16    $[[T1:[0-9]+]], 3
  ; MM32:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; MM32:       addu    $[[T2:[0-9]+]], $6, $[[T1]]
  ; MM32:       li16    $[[T3:[0-9]+]], 0
  ; MM32:       sltu    $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; MM32:       addu    $[[T4]], $5, $[[T4]]
  ; MM32:       sltu    $[[T5:[0-9]+]], $[[T4]], $[[T3]]
  ; MM32:       addu    $[[T5]], $4, $[[T5]]
  ; MM32:       move    $4, $[[T2]]
  ; MM32:       move    $5, $[[T0]]

  ; MM64:       daddiu  $[[T0:[0-9]+]], $5, 3
  ; MM64:       daddiu  $[[T1:[0-9]+]], $zero, 3
  ; MM64:       sltu    $[[T1]], $[[T0]], $[[T1]]
  ; MM64:       daddu   $2, $4, $[[T1]]

  %r = add i128 3, %a
  ret i128 %r
}
