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
; RUN: llc < %s -march=mips -mcpu=mips64r6 -target-abi n64 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM64

define signext i1 @xor_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: xor_i1:

  ; GP32:         xor     $2, $4, $5

  ; GP64:         xor     $2, $4, $5

  ; MM:           xor16   $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = xor i1 %a, %b
  ret i1 %r
}

define signext i8 @xor_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: xor_i8:

  ; GP32:         xor     $2, $4, $5

  ; GP64:         xor     $2, $4, $5

  ; MM:           xor16   $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = xor i8 %a, %b
  ret i8 %r
}

define signext i16 @xor_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: xor_i16:

  ; GP32:         xor     $2, $4, $5

  ; GP64:         xor     $2, $4, $5

  ; MM:           xor16   $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = xor i16 %a, %b
  ret i16 %r
}

define signext i32 @xor_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: xor_i32:

  ; GP32:         xor     $2, $4, $5

  ; GP64:         xor     $[[T0:[0-9]+]], $4, $5
  ; GP64:         sll     $2, $[[T0]], 0

  ; MM32:         xor16   $[[T0:[0-9]+]], $5
  ; MM32:         move    $2, $[[T0]]

  ; MM64:         xor     $[[T0:[0-9]+]], $4, $5
  ; MM64:         sll     $2, $[[T0]], 0

  %r = xor i32 %a, %b
  ret i32 %r
}

define signext i64 @xor_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: xor_i64:

  ; GP32:         xor     $2, $4, $6
  ; GP32:         xor     $3, $5, $7

  ; GP64:         xor     $2, $4, $5

  ; MM32:         xor16   $[[T0:[0-9]+]], $6
  ; MM32:         xor16   $[[T1:[0-9]+]], $7
  ; MM32:         move    $2, $[[T0]]
  ; MM32:         move    $3, $[[T1]]

  ; MM64:         xor     $2, $4, $5

  %r = xor i64 %a, %b
  ret i64 %r
}

define signext i128 @xor_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: xor_i128:

  ; GP32:         lw      $[[T0:[0-9]+]], 24($sp)
  ; GP32:         lw      $[[T1:[0-9]+]], 20($sp)
  ; GP32:         lw      $[[T2:[0-9]+]], 16($sp)
  ; GP32:         xor     $2, $4, $[[T2]]
  ; GP32:         xor     $3, $5, $[[T1]]
  ; GP32:         xor     $4, $6, $[[T0]]
  ; GP32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; GP32:         xor     $5, $7, $[[T3]]

  ; GP64:         xor     $2, $4, $6
  ; GP64:         xor     $3, $5, $7

  ; MM32:         lw      $[[T0:[0-9]+]], 32($sp)
  ; MM32:         lw      $[[T1:[0-9]+]], 28($sp)
  ; MM32:         lw      $[[T2:[0-9]+]], 24($sp)
  ; MM32:         xor16   $[[T2]], $4
  ; MM32:         xor16   $[[T1]], $5
  ; MM32:         xor16   $[[T0]], $6
  ; MM32:         lw      $[[T3:[0-9]+]], 36($sp)
  ; MM32:         xor16   $[[T3]], $7

  ; MM64:         xor     $2, $4, $6
  ; MM64:         xor     $3, $5, $7

  %r = xor i128 %a, %b
  ret i128 %r
}

define signext i1 @xor_i1_4(i1 signext %b) {
entry:
; ALL-LABEL: xor_i1_4:

  ; ALL:          move    $2, $4

  %r = xor i1 4, %b
  ret i1 %r
}

define signext i8 @xor_i8_4(i8 signext %b) {
entry:
; ALL-LABEL: xor_i8_4:

  ; ALL:          xori    $2, $4, 4

  %r = xor i8 4, %b
  ret i8 %r
}

define signext i16 @xor_i16_4(i16 signext %b) {
entry:
; ALL-LABEL: xor_i16_4:

  ; ALL:          xori    $2, $4, 4

  %r = xor i16 4, %b
  ret i16 %r
}

define signext i32 @xor_i32_4(i32 signext %b) {
entry:
; ALL-LABEL: xor_i32_4:

  ; ALL:          xori    $2, $4, 4

  %r = xor i32 4, %b
  ret i32 %r
}

define signext i64 @xor_i64_4(i64 signext %b) {
entry:
; ALL-LABEL: xor_i64_4:

  ; GP32:         xori    $3, $5, 4
  ; GP32:         move    $2, $4

  ; GP64:         xori    $2, $4, 4

  ; MM32:         xori    $3, $5, 4
  ; MM32:         move    $2, $4

  ; MM64:         xori    $2, $4, 4

  %r = xor i64 4, %b
  ret i64 %r
}

define signext i128 @xor_i128_4(i128 signext %b) {
entry:
; ALL-LABEL: xor_i128_4:

  ; GP32:         xori    $[[T0:[0-9]+]], $7, 4
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         xori    $3, $5, 4
  ; GP64:         move    $2, $4

  ; MM32:         xori    $[[T0:[0-9]+]], $7, 4
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         xori    $3, $5, 4
  ; MM64:         move    $2, $4

  %r = xor i128 4, %b
  ret i128 %r
}
