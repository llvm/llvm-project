; RUN: llc < %s -mtriple=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefixes=NOT-R2-R6,GP32,NOT-MM,PRE4
; RUN: llc < %s -mtriple=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefixes=NOT-R2-R6,GP32,NOT-MM
; RUN: llc < %s -mtriple=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP32,NOT-MM
; RUN: llc < %s -mtriple=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP32,NOT-MM
; RUN: llc < %s -mtriple=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP32,NOT-MM
; RUN: llc < %s -mtriple=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP32,NOT-MM
; RUN: llc < %s -mtriple=mips -mcpu=mips32r3 -mattr=+micromips -verify-machineinstrs | FileCheck %s \
; RUN:    -check-prefixes=MM32,MMR3
; RUN: llc < %s -mtriple=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=MM32,MMR6
; RUN: llc < %s -mtriple=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=NOT-R2-R6,GP64,NOT-MM,GP64-NOT-R2
; RUN: llc < %s -mtriple=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=NOT-R2-R6,GP64,NOT-MM,GP64-NOT-R2
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=NOT-R2-R6,GP64,NOT-MM,GP64-NOT-R2
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP64,NOT-MM,GP64-R2
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP64,NOT-MM,GP64-R2
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP64,NOT-MM,GP64-R2
; RUN: llc < %s -mtriple=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=R2-R6,GP64,NOT-MM,GP64-R2

define signext i1 @sub_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: sub_i1:

  ; NOT-MM:         subu    $[[T0:[0-9]+]], $4, $5
  ; NOT-MM:         andi    $[[T0]], $[[T0]], 1
  ; NOT-MM:         negu    $2, $[[T0]]

  ; MM:             subu16  $[[T0:[0-9]+]], $4, $5
  ; MM:             andi16  $[[T0]], $[[T0]], 1
  ; MM:             li16    $[[T1:[0-9]+]], 0
  ; MM:             subu16  $2, $[[T1]], $[[T0]]

  %r = sub i1 %a, %b
  ret i1 %r
}

define signext i8 @sub_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: sub_i8:

  ; NOT-R2-R6:      subu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:      sll     $[[T0]], $[[T0]], 24
  ; NOT-R2-R6:      sra     $2, $[[T0]], 24

  ; R2-R6:          subu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:          seb     $2, $[[T0:[0-9]+]]

  ; MM:             subu16  $[[T0:[0-9]+]], $4, $5
  ; MM:             seb     $[[T0]], $[[T0]]

  %r = sub i8 %a, %b
  ret i8 %r
}

define signext i16 @sub_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: sub_i16:

  ; NOT-R2-R6:      subu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:      sll     $[[T0]], $[[T0]], 16
  ; NOT-R2-R6:      sra     $2, $[[T0]], 16

  ; R2-R6:          subu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:          seh     $2, $[[T0:[0-9]+]]

  ; MM:             subu16  $[[T0:[0-9]+]], $4, $5
  ; MM:             seh     $[[T0]], $[[T0]]

  %r = sub i16 %a, %b
  ret i16 %r
}

define signext i32 @sub_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: sub_i32:

  ; NOT-MM:         subu    $2, $4, $5

  ; MM:             subu16  $2, $4, $5

  %r = sub i32 %a, %b
  ret i32 %r
}

define signext i64 @sub_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: sub_i64:

  ; GP32:    sltu    $[[T0:[0-9]+]], $5, $7
  ; GP32:    subu    $2, $4, $6
  ; GP32:    subu    $2, $2, $[[T0]]
  ; GP32:    subu    $3, $5, $7

  ; MM32:    sltu    $[[T0:[0-9]+]], $5, $7
  ; MM32:    subu16  $3, $4, $6
  ; MM32:    subu16  $2, $3, $[[T0]]
  ; MM32:    subu16  $3, $5, $7

  ; GP64:    dsubu   $2, $4, $5

  %r = sub i64 %a, %b
  ret i64 %r
}

define signext i128 @sub_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: sub_i128:

; PRE4: lw     $[[T0:[0-9]+]], 24($sp)
; PRE4: lw     $[[T1:[0-9]+]], 28($sp)
; PRE4: sltu   $[[T2:[0-9]+]], $7, $[[T1]]
; PRE4: xor    $[[T3:[0-9]+]], $6, $[[T0]]
; PRE4: sltiu  $[[T4:[0-9]+]], $[[T3]], 1
; PRE4: bnez   $[[T4]]
; PRE4: move   $[[T5:[0-9]+]], $[[T2]]
; PRE4: sltu   $[[T5]], $6, $[[T0]]

; PRE4: lw     $[[T6:[0-9]+]], 20($sp)
; PRE4: subu   $[[T7:[0-9]+]], $5, $[[T6]]
; PRE4: subu   $[[T8:[0-9]+]], $[[T7]], $[[T5]]
; PRE4: sltu   $[[T9:[0-9]+]], $[[T7]], $[[T5]]
; PRE4: sltu   $[[T10:[0-9]+]], $5, $[[T6]]
; PRE4: lw     $[[T11:[0-9]+]], 16($sp)
; PRE4: subu   $[[T12:[0-9]+]], $4, $[[T11]]
; PRE4: subu   $[[T13:[0-9]+]], $[[T12]], $[[T10]]
; PRE4: subu   $[[T14:[0-9]+]], $[[T13]], $[[T9]]
; PRE4: subu   $[[T15:[0-9]+]], $6, $[[T0]]
; PRE4: subu   $[[T16:[0-9]+]], $[[T15]], $[[T2]]
; PRE4: subu   $5, $7, $[[T1]]

; MMR3: lw       $[[T1:[0-9]+]], 48($sp)
; MMR3: sltu     $[[T2:[0-9]+]], $6, $[[T1]]
; MMR3: xor      $[[T3:[0-9]+]], $6, $[[T1]]
; MMR3: lw       $[[T4:[0-9]+]], 52($sp)
; MMR3: sltu     $[[T5:[0-9]+]], $7, $[[T4]]
; MMR3: movz     $[[T6:[0-9]+]], $[[T5]], $[[T3]]
; MMR3: lw       $[[T7:[0-8]+]], 44($sp)
; MMR3: subu16   $[[T8:[0-9]+]], $5, $[[T7]]
; MMR3: subu16   $[[T9:[0-9]+]], $[[T8]], $[[T6]]
; MMR3: sltu     $[[T10:[0-9]+]], $[[T8]], $[[T2]]
; MMR3: sltu     $[[T11:[0-9]+]], $5, $[[T7]]
; MMR3: lw       $[[T12:[0-9]+]], 40($sp)
; MMR3: lw       $[[T13:[0-9]+]], 12($sp)
; MMR3: subu16   $[[T14:[0-9]+]], $[[T13]], $[[T12]]
; MMR3: subu16   $[[T15:[0-9]+]], $[[T14]], $[[T11]]
; MMR3: subu16   $[[T16:[0-9]+]], $[[T15]], $[[T10]]
; MMR3: subu16   $[[T17:[0-9]+]], $6, $[[T1]]
; MMR3: subu16   $[[T18:[0-9]+]], $[[T17]], $7
; MMR3: lw       $[[T19:[0-9]+]], 8($sp)
; MMR3: lw       $[[T20:[0-9]+]], 0($sp)
; MMR3: subu16   $5, $[[T19]], $[[T20]]

; MMR6: move     $[[T0:[0-9]+]], $7
; MMR6: sw       $7, 8($sp)
; MMR6: move     $[[T1:[0-9]+]], $5
; MMR6: sw       $4, 12($sp)
; MMR6: lw       $[[T2:[0-9]+]], 48($sp)
; MMR6: sltu     $[[T3:[0-9]+]], $6, $[[T2]]
; MMR6: xor      $[[T4:[0-9]+]], $6, $[[T2]]
; MMR6: sltiu    $[[T5:[0-9]+]], $[[T4]], 1
; MMR6: seleqz   $[[T6:[0-9]+]], $[[T3]], $[[T5]]
; MMR6: lw       $[[T7:[0-9]+]], 52($sp)
; MMR6: sltu     $[[T8:[0-9]+]], $[[T0]], $[[T7]]
; MMR6: selnez   $[[T9:[0-9]+]], $[[T8]], $[[T5]]
; MMR6: or       $[[T10:[0-9]+]], $[[T9]], $[[T6]]
; MMR6: lw       $[[T11:[0-9]+]], 44($sp)
; MMR6: subu16   $[[T12:[0-9]+]], $[[T1]], $[[T11]]
; MMR6: subu16   $[[T13:[0-9]+]], $[[T12]], $[[T7]]
; MMR6: sltu     $[[T16:[0-9]+]], $[[T12]], $[[T7]]
; MMR6: sltu     $[[T17:[0-9]+]], $[[T1]], $[[T11]]
; MMR6: lw       $[[T18:[0-9]+]], 40($sp)
; MMR6: lw       $[[T19:[0-9]+]], 12($sp)
; MMR6: subu16   $[[T20:[0-9]+]], $[[T19]], $[[T18]]
; MMR6: subu16   $[[T21:[0-9]+]], $[[T20]], $[[T17]]
; MMR6: subu16   $[[T22:[0-9]+]], $[[T21]], $[[T16]]
; MMR6: subu16   $[[T23:[0-9]+]], $6, $[[T2]]
; MMR6: subu16   $4, $[[T23]], $5
; MMR6: lw       $[[T24:[0-9]+]], 8($sp)
; MMR6: lw       $[[T25:[0-9]+]], 0($sp)
; MMR6: subu16   $5, $[[T24]], $[[T25]]
; MMR6: lw       $3, 4($sp)

; FIXME: The sltu, dsll, dsrl pattern here occurs when an i32 is zero
;        extended to 64 bits. Fortunately slt(i)(u) actually gives an i1.
;        These should be combined away.

; GP64-NOT-R2: dsubu     $1, $4, $6
; GP64-NOT-R2: sltu      $[[T0:[0-9]+]], $5, $7
; GP64-NOT-R2: dsll      $[[T1:[0-9]+]], $[[T0]], 32
; GP64-NOT-R2: dsrl      $[[T2:[0-9]+]], $[[T1]], 32
; GP64-NOT-R2: dsubu     $2, $1, $[[T2]]
; GP64-NOT-R2: dsubu     $3, $5, $7

; FIXME: Likewise for the sltu, dext here.

; GP64-R2:     dsubu     $1, $4, $6
; GP64-R2:     sltu      $[[T0:[0-9]+]], $5, $7
; GP64-R2:     dext      $[[T1:[0-9]+]], $[[T0]], 0, 32
; GP64-R2:     dsubu     $2, $1, $[[T1]]
; GP64-R2:     dsubu     $3, $5, $7

  %r = sub i128 %a, %b
  ret i128 %r
}
