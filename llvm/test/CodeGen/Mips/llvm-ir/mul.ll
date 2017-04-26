; RUN: llc < %s -march=mips -mcpu=mips2 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,M2,GP32
; RUN: llc < %s -march=mips -mcpu=mips32 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1-R5,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1-R5,32R2-R5,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r3 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1-R5,32R2-R5,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r5 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1-R5,32R2-R5,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R6,GP32
; RUN: llc < %s -march=mips64 -mcpu=mips4 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,M4,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,64R1-R5,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,64R1-R5,GP64,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,64R1-R5,GP64,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,64R1-R5,GP64,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=ALL,64R6
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=MM32,MM32R3
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefixes=MM32,MM32R6
; RUN: llc < %s -march=mips -mcpu=mips64r6 -mattr=+micromips -target-abi n64 -relocation-model=pic | \
; RUN:   FileCheck %s -check-prefix=MM64R6

define signext i1 @mul_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: mul_i1:

  ; M2:         mult    $4, $5
  ; M2:         mflo    $[[T0:[0-9]+]]
  ; M2:         andi    $[[T0]], $[[T0]], 1
  ; M2:         negu    $2, $[[T0]]

  ; 32R1-R5:    mul     $[[T0:[0-9]+]], $4, $5
  ; 32R1-R5:    andi    $[[T0]], $[[T0]], 1
  ; 32R1-R5:    negu    $2, $[[T0]]

  ; 32R6:       mul     $[[T0:[0-9]+]], $4, $5
  ; 32R6:       andi    $[[T0]], $[[T0]], 1
  ; 32R6:       negu    $2, $[[T0]]

  ; M4:         mult    $4, $5
  ; M4:         mflo    $[[T0:[0-9]+]]
  ; M4:         andi    $[[T0]], $[[T0]], 1
  ; M4:         negu    $2, $[[T0]]

  ; 64R1-R5:    mul     $[[T0:[0-9]+]], $4, $5
  ; 64R1-R5:    andi    $[[T0]], $[[T0]], 1
  ; 64R1-R5:    negu    $2, $[[T0]]

  ; 64R6:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R6:       andi    $[[T0]], $[[T0]], 1
  ; 64R6:       negu    $2, $[[T0]]

  ; MM64R6:     mul     $[[T0:[0-9]+]], $4, $5
  ; MM64R6:     andi16  $[[T0]], $[[T0]], 1
  ; MM64R6:     li16    $[[T1:[0-9]+]], 0
  ; MM64R6:     subu16  $2, $[[T1]], $[[T0]]

  ; MM32:       mul     $[[T0:[0-9]+]], $4, $5
  ; MM32:       andi16  $[[T0]], $[[T0]], 1
  ; MM32:       li16    $[[T1:[0-9]+]], 0
  ; MM32:       subu16  $2, $[[T1]], $[[T0]]

  %r = mul i1 %a, %b
  ret i1 %r
}

define signext i8 @mul_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: mul_i8:

  ; M2:         mult    $4, $5
  ; M2:         mflo    $[[T0:[0-9]+]]
  ; M2:         sll     $[[T0]], $[[T0]], 24
  ; M2:         sra     $2, $[[T0]], 24

  ; 32R1:       mul     $[[T0:[0-9]+]], $4, $5
  ; 32R1:       sll     $[[T0]], $[[T0]], 24
  ; 32R1:       sra     $2, $[[T0]], 24

  ; 32R2-R5:    mul     $[[T0:[0-9]+]], $4, $5
  ; 32R2-R5:    seb     $2, $[[T0]]

  ; 32R6:       mul     $[[T0:[0-9]+]], $4, $5
  ; 32R6:       seb     $2, $[[T0]]

  ; M4:         mult    $4, $5
  ; M4:         mflo    $[[T0:[0-9]+]]
  ; M4:         sll     $[[T0]], $[[T0]], 24
  ; M4:         sra     $2, $[[T0]], 24

  ; 64R1:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R1:       sll     $[[T0]], $[[T0]], 24
  ; 64R1:       sra     $2, $[[T0]], 24

  ; 64R2:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R2:       seb     $2, $[[T0]]

  ; 64R6:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R6:       seb     $2, $[[T0]]

  ; MM64R6:     mul     $[[T0:[0-9]+]], $4, $5
  ; MM64R6:     seb     $2, $[[T0]]

  ; MM32:       mul     $[[T0:[0-9]+]], $4, $5
  ; MM32:       seb     $2, $[[T0]]

  %r = mul i8 %a, %b
  ret i8 %r
}

define signext i16 @mul_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: mul_i16:

  ; M2:         mult    $4, $5
  ; M2:         mflo    $[[T0:[0-9]+]]
  ; M2:         sll     $[[T0]], $[[T0]], 16
  ; M2:         sra     $2, $[[T0]], 16

  ; 32R1:       mul     $[[T0:[0-9]+]], $4, $5
  ; 32R1:       sll     $[[T0]], $[[T0]], 16
  ; 32R1:       sra     $2, $[[T0]], 16

  ; 32R2-R5:    mul     $[[T0:[0-9]+]], $4, $5
  ; 32R2-R5:    seh     $2, $[[T0]]

  ; 32R6:       mul     $[[T0:[0-9]+]], $4, $5
  ; 32R6:       seh     $2, $[[T0]]

  ; M4:         mult    $4, $5
  ; M4:         mflo    $[[T0:[0-9]+]]
  ; M4:         sll     $[[T0]], $[[T0]], 16
  ; M4:         sra     $2, $[[T0]], 16

  ; 64R1:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R1:       sll     $[[T0]], $[[T0]], 16
  ; 64R1:       sra     $2, $[[T0]], 16

  ; 64R2:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R2:       seh     $2, $[[T0]]

  ; 64R6:       mul     $[[T0:[0-9]+]], $4, $5
  ; 64R6:       seh     $2, $[[T0]]

  ; MM64R6:     mul     $[[T0:[0-9]+]], $4, $5
  ; MM64R6:     seh     $2, $[[T0]]

  ; MM32:       mul     $[[T0:[0-9]+]], $4, $5
  ; MM32:       seh     $2, $[[T0]]

  %r = mul i16 %a, %b
  ret i16 %r
}

define signext i32 @mul_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: mul_i32:

  ; M2:         mult    $4, $5
  ; M2:         mflo    $2

  ; 32R1-R5:    mul     $2, $4, $5
  ; 32R6:       mul     $2, $4, $5

  ; 64R1-R5:    mul     $2, $4, $5
  ; 64R6:       mul     $2, $4, $5
  ; MM64R6:     mul     $2, $4, $5

  ; MM32:       mul     $2, $4, $5

  %r = mul i32 %a, %b
  ret i32 %r
}

define signext i64 @mul_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: mul_i64:

  ; M2:         mult    $4, $7
  ; M2:         mflo    $[[T0:[0-9]+]]
  ; M2:         mult    $5, $6
  ; M2:         mflo    $[[T1:[0-9]+]]
  ; M2:         multu   $5, $7
  ; M2:         mflo    $3
  ; M2:         mfhi    $4
  ; M2:         addu    $[[T2:[0-9]+]], $4, $[[T1]]
  ; M2:         addu    $2, $[[T2]], $[[T0]]

  ; 32R1-R5:    multu   $5, $7
  ; 32R1-R5:    mflo    $3
  ; 32R1-R5:    mfhi    $[[T0:[0-9]+]]
  ; 32R1-R5:    mul     $[[T1:[0-9]+]], $5, $6
  ; 32R1-R5:    addu    $[[T0]], $[[T0]], $[[T1:[0-9]+]]
  ; 32R1-R5:    mul     $[[T2:[0-9]+]], $4, $7
  ; 32R1-R5:    addu    $2, $[[T0]], $[[T2]]

  ; 32R6-DAG:   mul     $[[T0:[0-9]+]], $5, $6
  ; 32R6:       muhu    $[[T1:[0-9]+]], $5, $7
  ; 32R6:       addu    $[[T0]], $[[T1]], $[[T0]]
  ; 32R6-DAG:   mul     $[[T2:[0-9]+]], $4, $7
  ; 32R6:       addu    $2, $[[T0]], $[[T2]]
  ; 32R6-DAG:   mul     $3, $5, $7

  ; M4:         dmult   $4, $5
  ; M4:         mflo    $2

  ; 64R1-R5:    dmult   $4, $5
  ; 64R1-R5:    mflo    $2

  ; 64R6:       dmul    $2, $4, $5
  ; MM64R6:     dmul    $2, $4, $5

  ; MM32R3:     multu   $[[T0:[0-9]+]], $7
  ; MM32R3:     mflo    $[[T1:[0-9]+]]
  ; MM32R3:     mfhi    $[[T2:[0-9]+]]
  ; MM32R3:     mul     $[[T0]], $[[T0]], $6
  ; MM32R3:     addu16  $2, $[[T2]], $[[T0]]
  ; MM32R3:     mul     $[[T3:[0-9]+]], $4, $7
  ; MM32R3:     addu16  $[[T2]], $[[T2]], $[[T3]]

  ; MM32R6:     mul     $[[T0:[0-9]+]], $5, $6
  ; MM32R6:     muhu    $[[T1:[0-9]+]], $5, $7
  ; MM32R6:     addu16  $[[T2:[0-9]+]], $[[T1]], $[[T0]]
  ; MM32R6:     mul     $[[T3:[0-9]+]], $4, $7
  ; MM32R6:     addu16  $2, $[[T2]], $[[T3]]
  ; MM32R6:     mul     $[[T1]], $5, $7

  %r = mul i64 %a, %b
  ret i64 %r
}

define signext i128 @mul_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: mul_i128:

  ; GP32:           lw      $25, %call16(__multi3)($gp)

  ; GP64-NOT-R6:    dmult   $4, $7
  ; GP64-NOT-R6:    mflo    $[[T0:[0-9]+]]
  ; GP64-NOT-R6:    dmult   $5, $6
  ; GP64-NOT-R6:    mflo    $[[T1:[0-9]+]]
  ; GP64-NOT-R6:    dmultu  $5, $7
  ; GP64-NOT-R6:    mflo    $3
  ; GP64-NOT-R6:    mfhi    $[[T2:[0-9]+]]
  ; GP64-NOT-R6:    daddu   $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; GP64-NOT-R6:    daddu   $2, $[[T3:[0-9]+]], $[[T0]]

  ; 64R6-DAG:       dmul    $[[T1:[0-9]+]], $5, $6
  ; 64R6:           dmuhu   $[[T2:[0-9]+]], $5, $7
  ; 64R6:           daddu   $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; 64R6-DAG:       dmul    $[[T0:[0-9]+]], $4, $7
  ; 64R6:           daddu   $2, $[[T1]], $[[T0]]
  ; 64R6-DAG:       dmul    $3, $5, $7

  ; MM64R6-DAG:     dmul    $[[T1:[0-9]+]], $5, $6
  ; MM64R6:         dmuhu   $[[T2:[0-9]+]], $5, $7
  ; MM64R6:         daddu   $[[T3:[0-9]+]], $[[T2]], $[[T1]]
  ; MM64R6-DAG:     dmul    $[[T0:[0-9]+]], $4, $7
  ; MM64R6:         daddu   $2, $[[T1]], $[[T0]]
  ; MM64R6-DAG:     dmul    $3, $5, $7

  ; MM32:           lw      $25, %call16(__multi3)($gp)

  %r = mul i128 %a, %b
  ret i128 %r
}
