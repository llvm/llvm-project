; RUN: llc < %s -march=mips -mcpu=mips2 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32,M2,NOT-R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32,NOT-R2-R6,32R1-R5
; RUN: llc < %s -march=mips -mcpu=mips32r2 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32,32R1-R5,R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r3 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32,32R1-R5,R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r5 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32,32R1-R5,R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r6 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32,32R6,R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips3 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,M3,NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,GP64-NOT-R6,NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,GP64-NOT-R6,NOT-R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,GP64-NOT-R6,R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,GP64-NOT-R6,R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,GP64-NOT-R6,R2-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64,64R6,R2-R6
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MMR3
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MMR6

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

  ; MM:         andi16  $[[T0:[0-9]+]], $5, 255
  ; MM:         sllv    $[[T1:[0-9]+]], $4, $[[T0]]
  ; MM:         seb     $2, $[[T1]]

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

  ; MM:         andi16  $[[T0:[0-9]+]], $5, 65535
  ; MM:         sllv    $[[T1:[0-9]+]], $4, $[[T0]]
  ; MM:         seh     $2, $[[T1]]

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

  ; 32R1-R5:    sllv      $[[T0:[0-9]+]], $4, $7
  ; 32R1-R5:    not       $[[T1:[0-9]+]], $7
  ; 32R1-R5:    srl       $[[T2:[0-9]+]], $5, 1
  ; 32R1-R5:    srlv      $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; 32R1-R5:    or        $2, $[[T0]], $[[T3]]
  ; 32R1-R5:    sllv      $[[T4:[0-9]+]], $5, $7
  ; 32R1-R5:    andi      $[[T5:[0-9]+]], $7, 32
  ; 32R1-R5:    movn      $2, $[[T4]], $[[T5]]
  ; 32R1-R5:    jr        $ra
  ; 32R1-R5:    movn      $3, $zero, $[[T5]]

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

  ; GP64:       dsllv     $2, $4, $5

  ; MMR3:       sllv      $[[T0:[0-9]+]], $4, $7
  ; MMR3:       srl16     $[[T1:[0-9]+]], $5, 1
  ; MMR3:       not16     $[[T2:[0-9]+]], $7
  ; MMR3:       srlv      $[[T3:[0-9]+]], $[[T1]], $[[T2]]
  ; MMR3:       or16      $[[T4:[0-9]+]], $[[T0]]
  ; MMR3:       sllv      $[[T5:[0-9]+]], $5, $7
  ; MMR3:       andi16    $[[T6:[0-9]+]], $7, 32
  ; MMR3:       movn      $[[T7:[0-9]+]], $[[T5]], $[[T6]]
  ; MMR3:       lui       $[[T8:[0-9]+]], 0
  ; MMR3:       movn      $3, $[[T8]], $[[T6]]

  ; MMR6:       sllv      $[[T0:[0-9]+]], $4, $7
  ; MMR6:       srl16     $[[T1:[0-9]+]], $5, 1
  ; MMR6:       not16     $[[T2:[0-9]+]], $7
  ; MMR6:       srlv      $[[T3:[0-9]+]], $[[T1]], $[[T2]]
  ; MMR6:       or16      $[[T4:[0-9]+]], $[[T0]]
  ; MMR6:       andi16    $[[T5:[0-9]+]], $7, 32
  ; MMR6:       seleqz    $[[T6:[0-9]+]], $[[T4]], $[[T5]]
  ; MMR6:       sllv      $[[T7:[0-9]+]], $5, $7
  ; MMR6:       selnez    $[[T8:[0-9]+]], $[[T7]], $[[T5]]
  ; MMR6:       or        $2, $[[T8]], $[[T6]]
  ; MMR6:       seleqz    $3, $[[T7]], $[[T5]]

  %r = shl i64 %a, %b
  ret i64 %r
}

define signext i128 @shl_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: shl_i128:

  ; GP32:           lw        $25, %call16(__ashlti3)($gp)

  ; M3:             sll       $[[T0:[0-9]+]], $7, 0
  ; M3:             dsllv     $[[T1:[0-9]+]], $5, $7
  ; M3:             andi      $[[T2:[0-9]+]], $[[T0]], 64
  ; M3:             bnez      $[[T3:[0-9]+]], [[BB0:\.LBB[0-9_]+]]
  ; M3:             move      $2, $[[T1]]
  ; M3:             dsllv     $[[T4:[0-9]+]], $4, $7
  ; M3:             dsrl      $[[T5:[0-9]+]], $5, 1
  ; M3:             not       $[[T6:[0-9]+]], $[[T0]]
  ; M3:             dsrlv     $[[T7:[0-9]+]], $[[T5]], $[[T6]]
  ; M3:             or        $2, $[[T4]], $[[T7]]
  ; M3:             [[BB0]]:
  ; M3:             bnez      $[[T3]], [[BB1:\.LBB[0-9_]+]]
  ; M3:             daddiu    $3, $zero, 0
  ; M3:             move      $3, $[[T1]]
  ; M3:             [[BB1]]:
  ; M3:             jr        $ra
  ; M3:             nop

  ; GP64-NOT-R6:    dsllv     $[[T0:[0-9]+]], $4, $7
  ; GP64-NOT-R6:    dsrl      $[[T1:[0-9]+]], $5, 1
  ; GP64-NOT-R6:    sll       $[[T2:[0-9]+]], $7, 0
  ; GP64-NOT-R6:    not       $[[T3:[0-9]+]], $[[T2]]
  ; GP64-NOT-R6:    dsrlv     $[[T4:[0-9]+]], $[[T1]], $[[T3]]
  ; GP64-NOT-R6:    or        $2, $[[T0]], $[[T4]]
  ; GP64-NOT-R6:    dsllv     $3, $5, $7
  ; GP64-NOT-R6:    andi      $[[T5:[0-9]+]], $[[T2]], 64
  ; GP64-NOT-R6:    movn      $2, $3, $[[T5]]
  ; GP64-NOT-R6:    jr        $ra
  ; GP64-NOT-R6:    movn      $3, $zero, $1

  ; 64R6:           dsllv     $[[T0:[0-9]+]], $4, $7
  ; 64R6:           dsrl      $[[T1:[0-9]+]], $5, 1
  ; 64R6:           sll       $[[T2:[0-9]+]], $7, 0
  ; 64R6:           not       $[[T3:[0-9]+]], $[[T2]]
  ; 64R6:           dsrlv     $[[T4:[0-9]+]], $[[T1]], $[[T3]]
  ; 64R6:           or        $[[T5:[0-9]+]], $[[T0]], $[[T4]]
  ; 64R6:           andi      $[[T6:[0-9]+]], $[[T2]], 64
  ; 64R6:           sll       $[[T7:[0-9]+]], $[[T6]], 0
  ; 64R6:           seleqz    $[[T8:[0-9]+]], $[[T5]], $[[T7]]
  ; 64R6:           dsllv     $[[T9:[0-9]+]], $5, $7
  ; 64R6:           selnez    $[[T10:[0-9]+]], $[[T9]], $[[T7]]
  ; 64R6:           or        $2, $[[T10]], $[[T8]]
  ; 64R6:           jr        $ra
  ; 64R6:           seleqz    $3, $[[T9]], $[[T7]]

  ; MM:             lw        $25, %call16(__ashlti3)($2)

  %r = shl i128 %a, %b
  ret i128 %r
}
