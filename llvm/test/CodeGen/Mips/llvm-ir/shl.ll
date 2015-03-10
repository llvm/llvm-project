; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 \
; RUN:    -check-prefix=M2 -check-prefix=NOT-R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 -check-prefix=NOT-R2-R6 \
; RUN:    -check-prefix=32R1-R2
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 \
; RUN:    -check-prefix=32R1-R2 -check-prefix=R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 \
; RUN:    -check-prefix=32R6 -check-prefix=R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP64 \
; RUN:    -check-prefix=M3 -check-prefix=NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP64 \
; RUN:    -check-prefix=GP64-NOT-R6 -check-prefix=NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP64 \
; RUN:    -check-prefix=GP64-NOT-R6 -check-prefix=NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP64 \
; RUN:    -check-prefix=GP64-NOT-R6 -check-prefix R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP64 \
; RUN:    -check-prefix=64R6 -check-prefix=R2-R6

define signext i1 @shl_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: shl_i1:

  ; ALL:        move    $2, $4

  %r = shl i1 %a, %b
  ret i1 %r
}

define signext i8 @shl_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: shl_i8:

  ; NOT-R2-R6:  andi    $[[T0:[0-9]+]], $5, 255
  ; NOT-R2-R6:  sllv    $[[T1:[0-9]+]], $4, $[[T0]]
  ; NOT-R2-R6:  sll     $[[T2:[0-9]+]], $[[T1]], 24
  ; NOT-R2-R6:  sra     $2, $[[T2]], 24

  ; R2-R6:      andi    $[[T0:[0-9]+]], $5, 255
  ; R2-R6:      sllv    $[[T1:[0-9]+]], $4, $[[T0]]
  ; R2-R6:      seb     $2, $[[T1]]

  %r = shl i8 %a, %b
  ret i8 %r
}

define signext i16 @shl_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: shl_i16:

  ; NOT-R2-R6:  andi    $[[T0:[0-9]+]], $5, 65535
  ; NOT-R2-R6:  sllv    $[[T1:[0-9]+]], $4, $[[T0]]
  ; NOT-R2-R6:  sll     $[[T2:[0-9]+]], $[[T1]], 16
  ; NOT-R2-R6:  sra     $2, $[[T2]], 16

  ; R2-R6:      andi    $[[T0:[0-9]+]], $5, 65535
  ; R2-R6:      sllv    $[[T1:[0-9]+]], $4, $[[T0]]
  ; R2-R6:      seh     $2, $[[T1]]

  %r = shl i16 %a, %b
  ret i16 %r
}

define signext i32 @shl_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: shl_i32:

  ; ALL:        sllv    $2, $4, $5

  %r = shl i32 %a, %b
  ret i32 %r
}

define signext i64 @shl_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: shl_i64:

  ; M2:         sllv      $[[T0:[0-9]+]], $5, $7
  ; M2:         andi      $[[T1:[0-9]+]], $7, 32
  ; M2:         bnez      $[[T1]], $[[BB0:BB[0-9_]+]]
  ; M2:         move      $2, $[[T0]]
  ; M2:         sllv      $[[T2:[0-9]+]], $4, $7
  ; M2:         not       $[[T3:[0-9]+]], $7
  ; M2:         srl       $[[T4:[0-9]+]], $5, 1
  ; M2:         srlv      $[[T5:[0-9]+]], $[[T4]], $[[T3]]
  ; M2:         or        $2, $[[T2]], $[[T3]]
  ; M2:         $[[BB0]]:
  ; M2:         bnez      $[[T1]], $[[BB1:BB[0-9_]+]]
  ; M2:         addiu     $3, $zero, 0
  ; M2:         move      $3, $[[T0]]
  ; M2:         $[[BB1]]:
  ; M2:         jr        $ra
  ; M2:         nop

  ; 32R1-R2:    sllv      $[[T0:[0-9]+]], $4, $7
  ; 32R1-R2:    not       $[[T1:[0-9]+]], $7
  ; 32R1-R2:    srl       $[[T2:[0-9]+]], $5, 1
  ; 32R1-R2:    srlv      $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; 32R1-R2:    or        $2, $[[T0]], $[[T3]]
  ; 32R1-R2:    sllv      $[[T4:[0-9]+]], $5, $7
  ; 32R1-R2:    andi      $[[T5:[0-9]+]], $7, 32
  ; 32R1-R2:    movn      $2, $[[T4]], $[[T5]]
  ; 32R1-R2:    jr        $ra
  ; 32R1-R2:    movn      $3, $zero, $[[T5]]

  ; 32R6:       sllv      $[[T0:[0-9]+]], $4, $7
  ; 32R6:       not       $[[T1:[0-9]+]], $7
  ; 32R6:       srl       $[[T2:[0-9]+]], $5, 1
  ; 32R6:       srlv      $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; 32R6:       or        $[[T4:[0-9]+]], $[[T0]], $[[T3]]
  ; 32R6:       andi      $[[T5:[0-9]+]], $7, 32
  ; 32R6:       seleqz    $[[T6:[0-9]+]], $[[T4]], $[[T2]]
  ; 32R6:       sllv      $[[T7:[0-9]+]], $5, $7
  ; 32R6:       selnez    $[[T8:[0-9]+]], $[[T7]], $[[T5]]
  ; 32R6:       or        $2, $[[T8]], $[[T6]]
  ; 32R6:       jr        $ra
  ; 32R6:       seleqz    $3, $[[T7]], $[[T5]]

  ; GP64:       sll       $[[T0:[0-9]+]], $5, 0
  ; GP64:       dsllv     $2, $4, $1

  %r = shl i64 %a, %b
  ret i64 %r
}
