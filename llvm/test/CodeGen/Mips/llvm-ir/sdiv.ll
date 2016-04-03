; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=NOT-R2-R6 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=NOT-R2-R6 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=R2-R5 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=R2-R5 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=R2-R5 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=R6 -check-prefix=GP32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=NOT-R2-R6 -check-prefix=GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=NOT-R2-R6 -check-prefix=GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=NOT-R2-R6 -check-prefix=GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=R2-R5 -check-prefix=GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=R2-R5 -check-prefix=GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefix=NOT-R6 -check-prefix=R2-R5 -check-prefix=GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefix=R6 -check-prefix=64R6

define signext i1 @sdiv_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: sdiv_i1:

  ; NOT-R6:       div     $zero, $4, $5
  ; NOT-R6:       teq     $5, $zero, 7
  ; NOT-R6:       mflo    $[[T0:[0-9]+]]
  ; FIXME: The sll/sra instructions are redundant since div is signed.
  ; NOT-R6:       sll     $[[T1:[0-9]+]], $[[T0]], 31
  ; NOT-R6:       sra     $2, $[[T1]], 31

  ; R6:           div     $[[T0:[0-9]+]], $4, $5
  ; R6:           teq     $5, $zero, 7
  ; FIXME: The sll/sra instructions are redundant since div is signed.
  ; R6:           sll     $[[T1:[0-9]+]], $[[T0]], 31
  ; R6:           sra     $2, $[[T1]], 31

  %r = sdiv i1 %a, %b
  ret i1 %r
}

define signext i8 @sdiv_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: sdiv_i8:

  ; NOT-R2-R6:    div     $zero, $4, $5
  ; NOT-R2-R6:    teq     $5, $zero, 7
  ; NOT-R2-R6:    mflo    $[[T0:[0-9]+]]
  ; FIXME: The sll/sra instructions are redundant since div is signed.
  ; NOT-R2-R6:    sll     $[[T1:[0-9]+]], $[[T0]], 24
  ; NOT-R2-R6:    sra     $2, $[[T1]], 24

  ; R2-R5:        div     $zero, $4, $5
  ; R2-R5:        teq     $5, $zero, 7
  ; R2-R5:        mflo    $[[T0:[0-9]+]]
  ; FIXME: This instruction is redundant.
  ; R2-R5:        seb     $2, $[[T0]]

  ; R6:           div     $[[T0:[0-9]+]], $4, $5
  ; R6:           teq     $5, $zero, 7
  ; FIXME: This instruction is redundant.
  ; R6:           seb     $2, $[[T0]]

  %r = sdiv i8 %a, %b
  ret i8 %r
}

define signext i16 @sdiv_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: sdiv_i16:

  ; NOT-R2-R6:    div     $zero, $4, $5
  ; NOT-R2-R6:    teq     $5, $zero, 7
  ; NOT-R2-R6:    mflo    $[[T0:[0-9]+]]
  ; FIXME: The sll/sra instructions are redundant since div is signed.
  ; NOT-R2-R6:    sll     $[[T1:[0-9]+]], $[[T0]], 16
  ; NOT-R2-R6:    sra     $2, $[[T1]], 16

  ; R2-R5:        div     $zero, $4, $5
  ; R2-R5:        teq     $5, $zero, 7
  ; R2-R5:        mflo    $[[T0:[0-9]+]]
  ; FIXME: This is instruction is redundant since div is signed.
  ; R2-R5:        seh     $2, $[[T0]]

  ; R6:           div     $[[T0:[0-9]+]], $4, $5
  ; R6:           teq     $5, $zero, 7
  ; FIXME: This is instruction is redundant since div is signed.
  ; R6:           seh     $2, $[[T0]]

  %r = sdiv i16 %a, %b
  ret i16 %r
}

define signext i32 @sdiv_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: sdiv_i32:

  ; NOT-R6:       div     $zero, $4, $5
  ; NOT-R6:       teq     $5, $zero, 7
  ; NOT-R6:       mflo    $2

  ; R6:           div     $2, $4, $5
  ; R6:           teq     $5, $zero, 7

  %r = sdiv i32 %a, %b
  ret i32 %r
}

define signext i64 @sdiv_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: sdiv_i64:

  ; GP32:         lw      $25, %call16(__divdi3)($gp)

  ; GP64-NOT-R6:  ddiv    $zero, $4, $5
  ; GP64-NOT-R6:  teq     $5, $zero, 7
  ; GP64-NOT-R6:  mflo    $2

  ; 64R6:         ddiv    $2, $4, $5
  ; 64R6:         teq     $5, $zero, 7

  %r = sdiv i64 %a, %b
  ret i64 %r
}

define signext i128 @sdiv_i128(i128 signext %a, i128 signext %b) {
entry:
  ; ALL-LABEL: sdiv_i128:

    ; GP32:         lw      $25, %call16(__divti3)($gp)

    ; GP64-NOT-R6:  ld      $25, %call16(__divti3)($gp)
    ; 64R6:         ld      $25, %call16(__divti3)($gp)

    %r = sdiv i128 %a, %b
    ret i128 %r
}
