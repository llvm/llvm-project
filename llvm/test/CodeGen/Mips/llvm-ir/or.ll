; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s -check-prefixes=ALL,GP64
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM32
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM32
; RUN: llc < %s -march=mips -mcpu=mips64r6 -target-abi n64 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM64

define signext i1 @or_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: or_i1:

  ; GP32:         or      $2, $4, $5

  ; GP64:         or      $2, $4, $5

  ; MM:           or16    $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = or i1 %a, %b
  ret i1 %r
}

define signext i8 @or_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: or_i8:

  ; GP32:         or      $2, $4, $5

  ; GP64:         or      $2, $4, $5

  ; MM:           or16    $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = or i8 %a, %b
  ret i8 %r
}

define signext i16 @or_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: or_i16:

  ; GP32:         or      $2, $4, $5

  ; GP64:         or      $2, $4, $5

  ; MM:           or16    $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = or i16 %a, %b
  ret i16 %r
}

define signext i32 @or_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: or_i32:

  ; GP32:         or      $2, $4, $5

  ; GP64:         or      $[[T0:[0-9]+]], $4, $5
  ; FIXME: The sll instruction below is redundant.
  ; GP64:         sll     $2, $[[T0]], 0

  ; MM32:         or16    $[[T0:[0-9]+]], $5
  ; MM32:         move    $2, $[[T0]]

  ; MM64:         or      $[[T0:[0-9]+]], $4, $5
  ; MM64:         sll     $2, $[[T0]], 0

  %r = or i32 %a, %b
  ret i32 %r
}

define signext i64 @or_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: or_i64:

  ; GP32:         or      $2, $4, $6
  ; GP32:         or      $3, $5, $7

  ; GP64:         or      $2, $4, $5

  ; MM32:         or16    $[[T0:[0-9]+]], $6
  ; MM32:         or16    $[[T1:[0-9]+]], $7
  ; MM32:         move    $2, $[[T0]]
  ; MM32:         move    $3, $[[T1]]

  ; MM64:         or      $2, $4, $5

  %r = or i64 %a, %b
  ret i64 %r
}

define signext i128 @or_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: or_i128:

  ; GP32:         lw      $[[T1:[0-9]+]], 20($sp)
  ; GP32:         lw      $[[T2:[0-9]+]], 16($sp)
  ; GP32:         or      $2, $4, $[[T2]]
  ; GP32:         or      $3, $5, $[[T1]]
  ; GP32:         lw      $[[T0:[0-9]+]], 24($sp)
  ; GP32:         or      $4, $6, $[[T0]]
  ; GP32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; GP32:         or      $5, $7, $[[T3]]

  ; GP64:         or      $2, $4, $6
  ; GP64:         or      $3, $5, $7

  ; MM32:         lw      $[[T1:[0-9]+]], 20($sp)
  ; MM32:         lw      $[[T2:[0-9]+]], 16($sp)
  ; MM32:         or16    $[[T2]], $4
  ; MM32:         or16    $[[T1]], $5
  ; MM32:         lw      $[[T0:[0-9]+]], 24($sp)
  ; MM32:         or16    $[[T0]], $6
  ; MM32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; MM32:         or16    $[[T3]], $7

  ; MM64:         or      $2, $4, $6
  ; MM64:         or      $3, $5, $7

  %r = or i128 %a, %b
  ret i128 %r
}

define signext i1 @or_i1_4(i1 signext %b) {
entry:
; ALL-LABEL: or_i1_4:

  ; ALL:          move    $2, $4

  %r = or i1 4, %b
  ret i1 %r
}

define signext i8 @or_i8_4(i8 signext %b) {
entry:
; ALL-LABEL: or_i8_4:

  ; ALL:          ori     $2, $4, 4

  %r = or i8 4, %b
  ret i8 %r
}

define signext i16 @or_i16_4(i16 signext %b) {
entry:
; ALL-LABEL: or_i16_4:

  ; ALL:          ori     $2, $4, 4

  %r = or i16 4, %b
  ret i16 %r
}

define signext i32 @or_i32_4(i32 signext %b) {
entry:
; ALL-LABEL: or_i32_4:

  ; ALL:          ori     $2, $4, 4

  %r = or i32 4, %b
  ret i32 %r
}

define signext i64 @or_i64_4(i64 signext %b) {
entry:
; ALL-LABEL: or_i64_4:

  ; GP32:         ori     $3, $5, 4
  ; GP32:         move    $2, $4

  ; GP64:         ori     $2, $4, 4

  ; MM32:         ori     $3, $5, 4
  ; MM32:         move    $2, $4

  ; MM64:         ori     $2, $4, 4

  %r = or i64 4, %b
  ret i64 %r
}

define signext i128 @or_i128_4(i128 signext %b) {
entry:
; ALL-LABEL: or_i128_4:

  ; GP32:         ori     $[[T0:[0-9]+]], $7, 4
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         ori     $3, $5, 4
  ; GP64:         move    $2, $4

  ; MM32:         ori     $[[T0:[0-9]+]], $7, 4
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         ori     $3, $5, 4
  ; MM64:         move    $2, $4

  %r = or i128 4, %b
  ret i128 %r
}

define signext i1 @or_i1_31(i1 signext %b) {
entry:
; ALL-LABEL: or_i1_31:

  ; GP32:         addiu   $2, $zero, -1

  ; GP64:         addiu   $2, $zero, -1

  ; MM:           li16    $2, -1

  %r = or i1 31, %b
  ret i1 %r
}

define signext i8 @or_i8_31(i8 signext %b) {
entry:
; ALL-LABEL: or_i8_31:

  ; ALL:          ori     $2, $4, 31

  %r = or i8 31, %b
  ret i8 %r
}

define signext i16 @or_i16_31(i16 signext %b) {
entry:
; ALL-LABEL: or_i16_31:

  ; ALL:          ori     $2, $4, 31

  %r = or i16 31, %b
  ret i16 %r
}

define signext i32 @or_i32_31(i32 signext %b) {
entry:
; ALL-LABEL: or_i32_31:

  ; ALL:          ori     $2, $4, 31

  %r = or i32 31, %b
  ret i32 %r
}

define signext i64 @or_i64_31(i64 signext %b) {
entry:
; ALL-LABEL: or_i64_31:

  ; GP32:         ori     $3, $5, 31
  ; GP32:         move    $2, $4

  ; GP64:         ori     $2, $4, 31

  ; MM32:         ori     $3, $5, 31
  ; MM32:         move    $2, $4

  ; MM64:         ori     $2, $4, 31

  %r = or i64 31, %b
  ret i64 %r
}

define signext i128 @or_i128_31(i128 signext %b) {
entry:
; ALL-LABEL: or_i128_31:

  ; GP32:         ori     $[[T0:[0-9]+]], $7, 31
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         ori     $3, $5, 31
  ; GP64:         move    $2, $4

  ; MM32:         ori     $[[T0:[0-9]+]], $7, 31
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         ori     $3, $5, 31
  ; MM64:         move    $2, $4

  %r = or i128 31, %b
  ret i128 %r
}

define signext i1 @or_i1_255(i1 signext %b) {
entry:
; ALL-LABEL: or_i1_255:

  ; GP32:         addiu   $2, $zero, -1

  ; GP64:         addiu   $2, $zero, -1

  ; MM:           li16    $2, -1

  %r = or i1 255, %b
  ret i1 %r
}

define signext i8 @or_i8_255(i8 signext %b) {
entry:
; ALL-LABEL: or_i8_255:

  ; GP32:         addiu   $2, $zero, -1

  ; GP64:         addiu   $2, $zero, -1

  ; MM:           li16    $2, -1

  %r = or i8 255, %b
  ret i8 %r
}

define signext i16 @or_i16_255(i16 signext %b) {
entry:
; ALL-LABEL: or_i16_255:

  ; ALL:          ori     $2, $4, 255

  %r = or i16 255, %b
  ret i16 %r
}

define signext i32 @or_i32_255(i32 signext %b) {
entry:
; ALL-LABEL: or_i32_255:

  ; ALL:          ori     $2, $4, 255

  %r = or i32 255, %b
  ret i32 %r
}

define signext i64 @or_i64_255(i64 signext %b) {
entry:
; ALL-LABEL: or_i64_255:

  ; GP32:         ori     $3, $5, 255
  ; GP32:         move    $2, $4

  ; GP64:         ori     $2, $4, 255

  ; MM32:         ori     $3, $5, 255
  ; MM32:         move    $2, $4

  ; MM64:         ori     $2, $4, 255

  %r = or i64 255, %b
  ret i64 %r
}

define signext i128 @or_i128_255(i128 signext %b) {
entry:
; ALL-LABEL: or_i128_255:

  ; GP32:         ori     $[[T0:[0-9]+]], $7, 255
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         ori     $3, $5, 255
  ; GP64:         move    $2, $4

  ; MM32:         ori     $[[T0:[0-9]+]], $7, 255
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         ori     $3, $5, 255
  ; MM64:         move    $2, $4

  %r = or i128 255, %b
  ret i128 %r
}

define signext i1 @or_i1_32768(i1 signext %b) {
entry:
; ALL-LABEL: or_i1_32768:

  ; ALL:          move    $2, $4

  %r = or i1 32768, %b
  ret i1 %r
}

define signext i8 @or_i8_32768(i8 signext %b) {
entry:
; ALL-LABEL: or_i8_32768:

  ; ALL:          move    $2, $4

  %r = or i8 32768, %b
  ret i8 %r
}

define signext i16 @or_i16_32768(i16 signext %b) {
entry:
; ALL-LABEL: or_i16_32768:

  ; GP32:         addiu   $[[T0:[0-9]+]], $zero, -32768
  ; GP32:         or      $2, $4, $[[T0]]

  ; GP64:         addiu   $[[T0:[0-9]+]], $zero, -32768
  ; GP64:         or      $2, $4, $[[T0]]

  ; MM:           addiu   $2, $zero, -32768
  ; MM:           or16    $2, $4

  %r = or i16 32768, %b
  ret i16 %r
}

define signext i32 @or_i32_32768(i32 signext %b) {
entry:
; ALL-LABEL: or_i32_32768:

  ; ALL:          ori     $2, $4, 32768

  %r = or i32 32768, %b
  ret i32 %r
}

define signext i64 @or_i64_32768(i64 signext %b) {
entry:
; ALL-LABEL: or_i64_32768:

  ; GP32:         ori     $3, $5, 32768
  ; GP32:         move    $2, $4

  ; GP64:         ori     $2, $4, 32768

  ; MM32:         ori     $3, $5, 32768
  ; MM32:         move    $2, $4

  ; MM64:         ori     $2, $4, 32768

  %r = or i64 32768, %b
  ret i64 %r
}

define signext i128 @or_i128_32768(i128 signext %b) {
entry:
; ALL-LABEL: or_i128_32768:

  ; GP32:         ori     $[[T0:[0-9]+]], $7, 32768
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         ori     $3, $5, 32768
  ; GP64:         move    $2, $4

  ; MM32:         ori     $[[T0:[0-9]+]], $7, 32768
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         ori     $3, $5, 32768
  ; MM64:         move    $2, $4

  %r = or i128 32768, %b
  ret i128 %r
}

define signext i1 @or_i1_65(i1 signext %b) {
entry:
; ALL-LABEL: or_i1_65:

  ; GP32:         addiu   $2, $zero, -1

  ; GP64:         addiu   $2, $zero, -1

  ; MM:           li16    $2, -1

  %r = or i1 65, %b
  ret i1 %r
}

define signext i8 @or_i8_65(i8 signext %b) {
entry:
; ALL-LABEL: or_i8_65:

  ; ALL:          ori     $2, $4, 65

  %r = or i8 65, %b
  ret i8 %r
}

define signext i16 @or_i16_65(i16 signext %b) {
entry:
; ALL-LABEL: or_i16_65:

  ; ALL:          ori     $2, $4, 65

  %r = or i16 65, %b
  ret i16 %r
}

define signext i32 @or_i32_65(i32 signext %b) {
entry:
; ALL-LABEL: or_i32_65:

  ; ALL:          ori     $2, $4, 65

  %r = or i32 65, %b
  ret i32 %r
}

define signext i64 @or_i64_65(i64 signext %b) {
entry:
; ALL-LABEL: or_i64_65:

  ; GP32:         ori     $3, $5, 65
  ; GP32:         move    $2, $4

  ; GP64:         ori     $2, $4, 65

  ; MM32:         ori     $3, $5, 65
  ; MM32:         move    $2, $4

  ; MM64:         ori     $2, $4, 65

  %r = or i64 65, %b
  ret i64 %r
}

define signext i128 @or_i128_65(i128 signext %b) {
entry:
; ALL-LABEL: or_i128_65:

  ; GP32:         ori     $[[T0:[0-9]+]], $7, 65
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         ori     $3, $5, 65
  ; GP64:         move    $2, $4

  ; MM32:         ori     $[[T0:[0-9]+]], $7, 65
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         ori     $3, $5, 65
  ; MM64:         move    $2, $4

  %r = or i128 65, %b
  ret i128 %r
}

define signext i1 @or_i1_256(i1 signext %b) {
entry:
; ALL-LABEL: or_i1_256:

  ; ALL:          move    $2, $4

  %r = or i1 256, %b
  ret i1 %r
}

define signext i8 @or_i8_256(i8 signext %b) {
entry:
; ALL-LABEL: or_i8_256:

  ; ALL:          move    $2, $4

  %r = or i8 256, %b
  ret i8 %r
}

define signext i16 @or_i16_256(i16 signext %b) {
entry:
; ALL-LABEL: or_i16_256:

  ; ALL:          ori     $2, $4, 256

  %r = or i16 256, %b
  ret i16 %r
}

define signext i32 @or_i32_256(i32 signext %b) {
entry:
; ALL-LABEL: or_i32_256:

  ; ALL:          ori     $2, $4, 256

  %r = or i32 256, %b
  ret i32 %r
}

define signext i64 @or_i64_256(i64 signext %b) {
entry:
; ALL-LABEL: or_i64_256:

  ; GP32:         ori     $3, $5, 256
  ; GP32:         move    $2, $4

  ; GP64:         ori     $2, $4, 256

  ; MM32:         ori     $3, $5, 256
  ; MM32:         move    $2, $4

  ; MM64:         ori     $2, $4, 256

  %r = or i64 256, %b
  ret i64 %r
}

define signext i128 @or_i128_256(i128 signext %b) {
entry:
; ALL-LABEL: or_i128_256:

  ; GP32:         ori     $[[T0:[0-9]+]], $7, 256
  ; GP32:         move    $2, $4
  ; GP32:         move    $3, $5
  ; GP32:         move    $4, $6
  ; GP32:         move    $5, $[[T0]]

  ; GP64:         ori     $3, $5, 256
  ; GP64:         move    $2, $4

  ; MM32:         ori     $[[T0:[0-9]+]], $7, 256
  ; MM32:         move    $2, $4
  ; MM32:         move    $3, $5
  ; MM32:         move    $4, $6
  ; MM32:         move    $5, $[[T0]]

  ; MM64:         ori     $3, $5, 256
  ; MM64:         move    $2, $4

  %r = or i128 256, %b
  ret i128 %r
}
