; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM32

define signext i1 @not_i1(i1 signext %a) {
entry:
; ALL-LABEL: not_i1:

  ; GP32:         not     $2, $4

  ; GP64:         not     $2, $4

  ; MM:           not16   $2, $4

  %r = xor i1 %a, -1
  ret i1 %r
}

define signext i8 @not_i8(i8 signext %a) {
entry:
; ALL-LABEL: not_i8:

  ; GP32:         not     $2, $4

  ; GP64:         not     $2, $4

  ; MM:           not16   $2, $4

  %r = xor i8 %a, -1
  ret i8 %r
}

define signext i16 @not_i16(i16 signext %a) {
entry:
; ALL-LABEL: not_i16:

  ; GP32:         not     $2, $4

  ; GP64:         not     $2, $4

  ; MM:           not16   $2, $4

  %r = xor i16 %a, -1
  ret i16 %r
}

define signext i32 @not_i32(i32 signext %a) {
entry:
; ALL-LABEL: not_i32:

  ; GP32:         not     $2, $4

  ; GP64:         not     $1, $4
  ; GP64:         sll     $2, $1, 0

  ; MM:           not16   $2, $4

  %r = xor i32 %a, -1
  ret i32 %r
}

define signext i64 @not_i64(i64 signext %a) {
entry:
; ALL-LABEL: not_i64:

  ; GP32:         not     $2, $4
  ; GP32:         not     $3, $5

  ; GP64:         daddiu  $[[T0:[0-9]+]], $zero, -1
  ; GP64:         xor     $2, $4, $[[T0]]

  ; MM32:         not16   $2, $4
  ; MM32:         not16   $3, $5

  %r = xor i64 %a, -1
  ret i64 %r
}

define signext i128 @not_i128(i128 signext %a) {
entry:
; ALL-LABEL: not_i128:

  ; GP32:         not     $2, $4
  ; GP32:         not     $3, $5
  ; GP32:         not     $4, $6
  ; GP32:         not     $5, $7

  ; GP64:         daddiu  $[[T0:[0-9]+]], $zero, -1
  ; GP64:         xor     $2, $4, $[[T0]]
  ; GP64:         xor     $3, $5, $[[T0]]

  ; MM32:         not16   $2, $4
  ; MM32:         not16   $3, $5
  ; MM32:         not16   $4, $6
  ; MM32:         not16   $5, $7

  %r = xor i128 %a, -1
  ret i128 %r
}

define signext i1 @nor_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: nor_i1:

  ; GP32:         nor     $2, $5, $4
  ; GP64:         or      $1, $5, $4
  ; MM32:         nor     $2, $5, $4

  %or = or i1 %b, %a
  %r = xor i1 %or, -1
  ret i1 %r
}

define signext i8 @nor_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: nor_i8:

  ; GP32:         nor     $2, $5, $4
  ; GP64:         or      $1, $5, $4
  ; MM32:         nor     $2, $5, $4

  %or = or i8 %b, %a
  %r = xor i8 %or, -1
  ret i8 %r
}

define signext i16 @nor_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: nor_i16:

  ; GP32:         nor     $2, $5, $4
  ; GP64:         or      $1, $5, $4
  ; MM32:         nor     $2, $5, $4

  %or = or i16 %b, %a
  %r = xor i16 %or, -1
  ret i16 %r
}

define signext i32 @nor_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: nor_i32:

  ; GP32:         nor     $2, $5, $4

  ; GP64:         or      $[[T0:[0-9]+]], $5, $4
  ; GP64:         sll     $[[T1:[0-9]+]], $[[T0]], 0
  ; GP64:         not     $[[T2:[0-9]+]], $[[T1]]
  ; GP64:         sll     $2, $[[T2]], 0

  ; MM32:         nor     $2, $5, $4

  %or = or i32 %b, %a
  %r = xor i32 %or, -1
  ret i32 %r
}


define signext i64 @nor_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: nor_i64:

  ; GP32:         nor     $2, $6, $4
  ; GP32:         nor     $3, $7, $5

  ; GP64:         nor     $2, $5, $4

  ; MM32:         nor     $2, $6, $4
  ; MM32:         nor     $3, $7, $5

  %or = or i64 %b, %a
  %r = xor i64 %or, -1
  ret i64 %r
}

define signext i128 @nor_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: nor_i128:

  ; GP32:         lw      $[[T1:[0-9]+]], 20($sp)
  ; GP32:         lw      $[[T2:[0-9]+]], 16($sp)
  ; GP32:         nor     $2, $[[T2]], $4
  ; GP32:         nor     $3, $[[T1]], $5
  ; GP32:         lw      $[[T0:[0-9]+]], 24($sp)
  ; GP32:         nor     $4, $[[T0]], $6
  ; GP32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; GP32:         nor     $5, $[[T3]], $7

  ; GP64:         nor     $2, $6, $4
  ; GP64:         nor     $3, $7, $5

  ; MM32:         lw      $[[T1:[0-9]+]], 20($sp)
  ; MM32:         lw      $[[T2:[0-9]+]], 16($sp)
  ; MM32:         nor     $2, $[[T2]], $4
  ; MM32:         nor     $3, $[[T1]], $5
  ; MM32:         lw      $[[T0:[0-9]+]], 24($sp)
  ; MM32:         nor     $4, $[[T0]], $6
  ; MM32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; MM32:         nor     $5, $[[T3]], $7

  %or = or i128 %b, %a
  %r = xor i128 %or, -1
  ret i128 %r
}
