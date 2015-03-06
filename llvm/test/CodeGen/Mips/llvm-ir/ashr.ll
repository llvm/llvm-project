; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 -check-prefix=NOT-R2-R6 \
; RUN:    -check-prefix=32R1-R2
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 \
; RUN:    -check-prefix=32R1-R2 -check-prefix=R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=GP32 \
; RUN:    -check-prefix=32R6 -check-prefix=R2-R6
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

define signext i1 @ashr_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: ashr_i1:

  ; ALL:        move    $2, $4

  %r = ashr i1 %a, %b
  ret i1 %r
}

define signext i8 @ashr_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: ashr_i8:

  ; FIXME: The andi instruction is redundant.
  ; ALL:        andi    $[[T0:[0-9]+]], $5, 255
  ; ALL:        srav    $2, $4, $[[T0]]

  %r = ashr i8 %a, %b
  ret i8 %r
}

define signext i16 @ashr_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: ashr_i16:

  ; FIXME: The andi instruction is redundant.
  ; ALL:        andi    $[[T0:[0-9]+]], $5, 65535
  ; ALL:        srav    $2, $4, $[[T0]]

  %r = ashr i16 %a, %b
  ret i16 %r
}

define signext i32 @ashr_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: ashr_i32:

  ; ALL:        srav    $2, $4, $5

  %r = ashr i32 %a, %b
  ret i32 %r
}

define signext i64 @ashr_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: ashr_i64:

  ; 32R1-R2:    srlv      $[[T0:[0-9]+]], $5, $7
  ; 32R1-R2:    not       $[[T1:[0-9]+]], $7
  ; 32R1-R2:    sll       $[[T2:[0-9]+]], $4, 1
  ; 32R1-R2:    sllv      $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; 32R1-R2:    or        $3, $[[T3]], $[[T0]]
  ; 32R1-R2:    srav      $[[T4:[0-9]+]], $4, $7
  ; 32R1-R2:    andi      $[[T5:[0-9]+]], $7, 32
  ; 32R1-R2:    movn      $3, $[[T4]], $[[T5]]
  ; 32R1-R2:    sra       $4, $4, 31
  ; 32R1-R2:    jr        $ra
  ; 32R1-R2:    movn      $2, $4, $[[T5]]

  ; 32R6:       srav      $[[T0:[0-9]+]], $4, $7
  ; 32R6:       andi      $[[T1:[0-9]+]], $7, 32
  ; 32R6:       seleqz    $[[T2:[0-9]+]], $[[T0]], $[[T1]]
  ; 32R6:       sra       $[[T3:[0-9]+]], $4, 31
  ; 32R6:       selnez    $[[T4:[0-9]+]], $[[T3]], $[[T1]]
  ; 32R6:       or        $[[T5:[0-9]+]], $[[T4]], $[[T2]]
  ; 32R6:       srlv      $[[T6:[0-9]+]], $5, $7
  ; 32R6:       not       $[[T7:[0-9]+]], $7
  ; 32R6:       sll       $[[T8:[0-9]+]], $4, 1
  ; 32R6:       sllv      $[[T9:[0-9]+]], $[[T8]], $[[T7]]
  ; 32R6:       or        $[[T10:[0-9]+]], $[[T9]], $[[T6]]
  ; 32R6:       seleqz    $[[T11:[0-9]+]], $[[T10]], $[[T1]]
  ; 32R6:       selnez    $[[T12:[0-9]+]], $[[T0]], $[[T1]]
  ; 32R6:       jr        $ra
  ; 32R6:       or        $3, $[[T0]], $[[T11]]

  ; FIXME: The sll instruction below is redundant.
  ; GP64:       sll       $[[T0:[0-9]+]], $5, 0
  ; GP64:       dsrav     $2, $4, $[[T0]]

  %r = ashr i64 %a, %b
  ret i64 %r
}

define signext i128 @ashr_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: ashr_i128:

  ; GP32:           lw        $25, %call16(__ashrti3)($gp)

  ; M3:             sll       $[[T0:[0-9]+]], $7, 0
  ; M3:             dsrav     $[[T1:[0-9]+]], $4, $[[T0]]
  ; M3:             andi      $[[T2:[0-9]+]], $[[T0]], 32
  ; M3:             bnez      $[[T3:[0-9]+]], $[[BB0:BB[0-9_]+]]
  ; M3:             move      $3, $[[T1]]
  ; M3:             dsrlv     $[[T4:[0-9]+]], $5, $[[T0]]
  ; M3:             dsll      $[[T5:[0-9]+]], $4, 1
  ; M3:             not       $[[T6:[0-9]+]], $[[T0]]
  ; M3:             dsllv     $[[T7:[0-9]+]], $[[T5]], $[[T6]]
  ; M3:             or        $3, $[[T7]], $[[T4]]
  ; M3:             $[[BB0]]:
  ; M3:             beqz      $[[T3]], $[[BB1:BB[0-9_]+]]
  ; M3:             nop
  ; M3:             dsra      $2, $4, 31
  ; M3:             $[[BB1]]:
  ; M3:             jr        $ra
  ; M3:             nop

  ; GP64-NOT-R6:    sll       $[[T0:[0-9]+]], $7, 0
  ; GP64-NOT-R6:    dsrlv     $[[T1:[0-9]+]], $5, $[[T0]]
  ; GP64-NOT-R6:    dsll      $[[T2:[0-9]+]], $4, 1
  ; GP64-NOT-R6:    not       $[[T3:[0-9]+]], $[[T0]]
  ; GP64-NOT-R6:    dsllv     $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; GP64-NOT-R6:    or        $3, $[[T4]], $[[T1]]
  ; GP64-NOT-R6:    dsrav     $2, $4, $[[T0]]
  ; GP64-NOT-R6:    andi      $[[T5:[0-9]+]], $[[T0]], 32

  ; GP64-NOT-R6:    movn      $3, $2, $[[T5]]
  ; GP64-NOT-R6:    dsra      $[[T6:[0-9]+]], $4, 31
  ; GP64-NOT-R6:    jr        $ra
  ; GP64-NOT-R6:    movn      $2, $[[T6]], $[[T5]]

  ; 64R6:           sll       $[[T0:[0-9]+]], $7, 0
  ; 64R6:           dsrav     $[[T1:[0-9]+]], $4, $[[T0]]
  ; 64R6:           andi      $[[T2:[0-9]+]], $[[T0]], 32
  ; 64R6:           sll       $[[T3:[0-9]+]], $[[T2]], 0
  ; 64R6:           seleqz    $[[T4:[0-9]+]], $[[T1]], $[[T3]]
  ; 64R6:           dsra      $[[T5:[0-9]+]], $4, 31
  ; 64R6:           selnez    $[[T6:[0-9]+]], $[[T5]], $[[T3]]
  ; 64R6:           or        $2, $[[T6]], $[[T4]]
  ; 64R6:           dsrlv     $[[T7:[0-9]+]], $5, $[[T0]]
  ; 64R6:           dsll      $[[T8:[0-9]+]], $4, 1
  ; 64R6:           not       $[[T9:[0-9]+]], $[[T0]]
  ; 64R6:           dsllv     $[[T10:[0-9]+]], $[[T8]], $[[T9]]
  ; 64R6:           or        $[[T11:[0-9]+]], $[[T10]], $[[T7]]
  ; 64R6:           seleqz    $[[T12:[0-9]+]], $[[T11]], $[[T3]]
  ; 64R6:           selnez    $[[T13:[0-9]+]], $[[T1]], $[[T3]]
  ; 64R6:           jr        $ra
  ; 64R6:           or        $3, $[[T13]], $[[T12]]

  %r = ashr i128 %a, %b
  ret i128 %r
}
