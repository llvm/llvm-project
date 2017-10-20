; RUN: llc < %s -march=mips -mcpu=mips2 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r3 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r5 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,R6,GP32

; RUN: llc < %s -march=mips64 -mcpu=mips3 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,NOT-R6,GP64-NOT-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,R6,64R6

; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR3,MM32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM32
; RUN: llc < %s -march=mips -mcpu=mips64r6 -target-abi n64 -mattr=+micromips -relocation-model=pic | FileCheck %s \
; RUN:    -check-prefixes=ALL,MMR6,MM64

define zeroext i1 @udiv_i1(i1 zeroext %a, i1 zeroext %b) {
entry:
; ALL-LABEL: udiv_i1:

  ; NOT-R6:       divu    $zero, $4, $5
  ; NOT-R6:       teq     $5, $zero, 7
  ; NOT-R6:       mflo    $2

  ; R6:           divu    $2, $4, $5
  ; R6:           teq     $5, $zero, 7

  ; MMR3:         divu    $zero, $4, $5
  ; MMR3:         teq     $5, $zero, 7
  ; MMR3:         mflo    $2

  ; MMR6:         divu    $2, $4, $5
  ; MMR6:         teq     $5, $zero, 7

  %r = udiv i1 %a, %b
  ret i1 %r
}

define zeroext i8 @udiv_i8(i8 zeroext %a, i8 zeroext %b) {
entry:
; ALL-LABEL: udiv_i8:

  ; NOT-R6:       divu    $zero, $4, $5
  ; NOT-R6:       teq     $5, $zero, 7
  ; NOT-R6:       mflo    $2

  ; R6:           divu    $2, $4, $5
  ; R6:           teq     $5, $zero, 7

  ; MMR3:         divu    $zero, $4, $5
  ; MMR3:         teq     $5, $zero, 7
  ; MMR3:         mflo    $2

  ; MMR6:         divu    $2, $4, $5
  ; MMR6:         teq     $5, $zero, 7

  %r = udiv i8 %a, %b
  ret i8 %r
}

define zeroext i16 @udiv_i16(i16 zeroext %a, i16 zeroext %b) {
entry:
; ALL-LABEL: udiv_i16:

  ; NOT-R6:       divu    $zero, $4, $5
  ; NOT-R6:       teq     $5, $zero, 7
  ; NOT-R6:       mflo    $2

  ; R6:           divu    $2, $4, $5
  ; R6:           teq     $5, $zero, 7

  ; MMR3:         divu    $zero, $4, $5
  ; MMR3:         teq     $5, $zero, 7
  ; MMR3:         mflo    $2

  ; MMR6:         divu    $2, $4, $5
  ; MMR6:         teq     $5, $zero, 7

  %r = udiv i16 %a, %b
  ret i16 %r
}

define signext i32 @udiv_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: udiv_i32:

  ; NOT-R6:       divu    $zero, $4, $5
  ; NOT-R6:       teq     $5, $zero, 7
  ; NOT-R6:       mflo    $2

  ; R6:           divu    $2, $4, $5
  ; R6:           teq     $5, $zero, 7

  ; MMR3:         divu    $zero, $4, $5
  ; MMR3:         teq     $5, $zero, 7
  ; MMR3:         mflo    $2

  ; MMR6:         divu    $2, $4, $5
  ; MMR6:         teq     $5, $zero, 7

  %r = udiv i32 %a, %b
  ret i32 %r
}

define signext i64 @udiv_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: udiv_i64:

  ; GP32:         lw      $25, %call16(__udivdi3)($gp)

  ; GP64-NOT-R6:  ddivu   $zero, $4, $5
  ; GP64-NOT-R6:  teq     $5, $zero, 7
  ; GP64-NOT-R6:  mflo    $2

  ; 64R6:         ddivu   $2, $4, $5
  ; 64R6:         teq     $5, $zero, 7

  ; MM32:         lw      $25, %call16(__udivdi3)($2)

  ; MM64:         ddivu   $2, $4, $5
  ; MM64:         teq     $5, $zero, 7

  %r = udiv i64 %a, %b
  ret i64 %r
}

define signext i128 @udiv_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: udiv_i128:

  ; GP32:         lw      $25, %call16(__udivti3)($gp)

  ; GP64-NOT-R6:  ld      $25, %call16(__udivti3)($gp)
  ; 64-R6:        ld      $25, %call16(__udivti3)($gp)

  ; MM32:         lw      $25, %call16(__udivti3)($16)

  ; MM64:         ld      $25, %call16(__udivti3)($2)

  %r = udiv i128 %a, %b
  ret i128 %r
}
