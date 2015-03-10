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

  ; M2:         srav      $[[T0:[0-9]+]], $4, $7
  ; M2:         andi      $[[T1:[0-9]+]], $7, 32
  ; M2:         bnez      $[[T1]], $[[BB0:BB[0-9_]+]]
  ; M2:         move      $3, $[[T0]]
  ; M2:         srlv      $[[T2:[0-9]+]], $5, $7
  ; M2:         not       $[[T3:[0-9]+]], $7
  ; M2:         sll       $[[T4:[0-9]+]], $4, 1
  ; M2:         sllv      $[[T5:[0-9]+]], $[[T4]], $[[T3]]
  ; M2:         or        $3, $[[T3]], $[[T2]]
  ; M2:         $[[BB0]]:
  ; M2:         beqz      $[[T1]], $[[BB1:BB[0-9_]+]]
  ; M2:         nop
  ; M2:         sra       $2, $4, 31
  ; M2:         $[[BB1]]:
  ; M2:         jr        $ra
  ; M2:         nop

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
