; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefixes=ALL,GP32
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

define signext i1 @and_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: and_i1:

  ; GP32:         and     $2, $4, $5

  ; GP64:         and     $2, $4, $5

  ; MM:           and16   $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = and i1 %a, %b
  ret i1 %r
}

define signext i8 @and_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: and_i8:

  ; GP32:         and     $2, $4, $5

  ; GP64:         and     $2, $4, $5

  ; MM:           and16   $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = and i8 %a, %b
  ret i8 %r
}

define signext i16 @and_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: and_i16:

  ; GP32:         and     $2, $4, $5

  ; GP64:         and     $2, $4, $5

  ; MM:           and16   $[[T0:[0-9]+]], $5
  ; MM:           move    $2, $[[T0]]

  %r = and i16 %a, %b
  ret i16 %r
}

define signext i32 @and_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: and_i32:

  ; GP32:         and     $2, $4, $5

  ; GP64:         and     $[[T0:[0-9]+]], $4, $5
  ; GP64:         sll     $2, $[[T0]], 0

  ; MM32:         and16   $[[T0:[0-9]+]], $5
  ; MM32:         move    $2, $[[T0]]

  ; MM64:         and     $[[T0:[0-9]+]], $4, $5
  ; MM64:         sll     $2, $[[T0]], 0

  %r = and i32 %a, %b
  ret i32 %r
}

define signext i64 @and_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: and_i64:

  ; GP32:         and     $2, $4, $6
  ; GP32:         and     $3, $5, $7

  ; GP64:         and     $2, $4, $5

  ; MM32:         and16   $[[T0:[0-9]+]], $6
  ; MM32:         and16   $[[T1:[0-9]+]], $7
  ; MM32:         move    $2, $[[T0]]
  ; MM32:         move    $3, $[[T1]]

  ; MM64:         and     $2, $4, $5

  %r = and i64 %a, %b
  ret i64 %r
}

define signext i128 @and_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: and_i128:

  ; GP32:         lw      $[[T0:[0-9]+]], 24($sp)
  ; GP32:         lw      $[[T1:[0-9]+]], 20($sp)
  ; GP32:         lw      $[[T2:[0-9]+]], 16($sp)
  ; GP32:         and     $2, $4, $[[T2]]
  ; GP32:         and     $3, $5, $[[T1]]
  ; GP32:         and     $4, $6, $[[T0]]
  ; GP32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; GP32:         and     $5, $7, $[[T3]]

  ; GP64:         and     $2, $4, $6
  ; GP64:         and     $3, $5, $7

  ; MM32:         lw      $[[T0:[0-9]+]], 20($sp)
  ; MM32:         lw      $[[T1:[0-9]+]], 16($sp)
  ; MM32:         and16   $[[T1]], $4
  ; MM32:         and16   $[[T0]], $5
  ; MM32:         lw      $[[T2:[0-9]+]], 24($sp)
  ; MM32:         and16   $[[T2]], $6
  ; MM32:         lw      $[[T3:[0-9]+]], 28($sp)
  ; MM32:         and16   $[[T3]], $7

  ; MM64:         and     $2, $4, $6
  ; MM64:         and     $3, $5, $7

  %r = and i128 %a, %b
  ret i128 %r
}

define signext i1 @and_i1_4(i1 signext %b) {
entry:
; ALL-LABEL: and_i1_4:

  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         addiu   $2, $zero, 0

  ; MM:           li16     $2, 0

  %r = and i1 4, %b
  ret i1 %r
}

define signext i8 @and_i8_4(i8 signext %b) {
entry:
; ALL-LABEL: and_i8_4:

  ; GP32:         andi    $2, $4, 4

  ; GP64:         andi    $2, $4, 4

  ; MM:           andi16  $2, $4, 4

  %r = and i8 4, %b
  ret i8 %r
}

define signext i16 @and_i16_4(i16 signext %b) {
entry:
; ALL-LABEL: and_i16_4:

  ; GP32:         andi    $2, $4, 4

  ; GP64:         andi    $2, $4, 4

  ; MM:           andi16  $2, $4, 4

  %r = and i16 4, %b
  ret i16 %r
}

define signext i32 @and_i32_4(i32 signext %b) {
entry:
; ALL-LABEL: and_i32_4:

  ; GP32:         andi    $2, $4, 4

  ; GP64:         andi    $2, $4, 4

  ; MM:           andi16  $2, $4, 4

  %r = and i32 4, %b
  ret i32 %r
}

define signext i64 @and_i64_4(i64 signext %b) {
entry:
; ALL-LABEL: and_i64_4:

  ; GP32:         andi    $3, $5, 4
  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         andi    $2, $4, 4

  ; MM32:         andi16  $3, $5, 4
  ; MM32:         li16     $2, 0

  ; MM64:         andi    $2, $4, 4

  %r = and i64 4, %b
  ret i64 %r
}

define signext i128 @and_i128_4(i128 signext %b) {
entry:
; ALL-LABEL: and_i128_4:

  ; GP32:         andi    $5, $7, 4
  ; GP32:         addiu   $2, $zero, 0
  ; GP32:         addiu   $3, $zero, 0
  ; GP32:         addiu   $4, $zero, 0

  ; GP64:         andi    $3, $5, 4
  ; GP64:         daddiu  $2, $zero, 0

  ; MM32:         andi16  $5, $7, 4
  ; MM32:         li16    $2, 0
  ; MM32:         li16    $3, 0
  ; MM32:         li16    $4, 0

  ; MM64:         andi    $3, $5, 4
  ; MM64:         daddiu  $2, $zero, 0

  %r = and i128 4, %b
  ret i128 %r
}

define signext i1 @and_i1_31(i1 signext %b) {
entry:
; ALL-LABEL: and_i1_31:

  ; ALL:          move    $2, $4

  %r = and i1 31, %b
  ret i1 %r
}

define signext i8 @and_i8_31(i8 signext %b) {
entry:
; ALL-LABEL: and_i8_31:

  ; GP32:         andi    $2, $4, 31

  ; GP64:         andi    $2, $4, 31

  ; MM:           andi16  $2, $4, 31

  %r = and i8 31, %b
  ret i8 %r
}

define signext i16 @and_i16_31(i16 signext %b) {
entry:
; ALL-LABEL: and_i16_31:

  ; GP32:         andi    $2, $4, 31

  ; GP64:         andi    $2, $4, 31

  ; MM:           andi16  $2, $4, 31

  %r = and i16 31, %b
  ret i16 %r
}

define signext i32 @and_i32_31(i32 signext %b) {
entry:
; ALL-LABEL: and_i32_31:

  ; GP32:         andi    $2, $4, 31

  ; GP64:         andi    $2, $4, 31

  ; MM:           andi16  $2, $4, 31

  %r = and i32 31, %b
  ret i32 %r
}

define signext i64 @and_i64_31(i64 signext %b) {
entry:
; ALL-LABEL: and_i64_31:

  ; GP32:         andi    $3, $5, 31
  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         andi    $2, $4, 31

  ; MM32:         andi16  $3, $5, 31
  ; MM32:         li16    $2, 0

  ; MM64:         andi    $2, $4, 31

  %r = and i64 31, %b
  ret i64 %r
}

define signext i128 @and_i128_31(i128 signext %b) {
entry:
; ALL-LABEL: and_i128_31:

  ; GP32:         andi    $5, $7, 31
  ; GP32:         addiu   $2, $zero, 0
  ; GP32:         addiu   $3, $zero, 0
  ; GP32:         addiu   $4, $zero, 0

  ; GP64:         andi    $3, $5, 31
  ; GP64:         daddiu  $2, $zero, 0

  ; MM32:         andi16  $5, $7, 31
  ; MM32:         li16    $2, 0
  ; MM32:         li16    $3, 0
  ; MM32:         li16    $4, 0

  ; MM64:         andi    $3, $5, 31
  ; MM64:         daddiu  $2, $zero, 0

  %r = and i128 31, %b
  ret i128 %r
}

define signext i1 @and_i1_255(i1 signext %b) {
entry:
; ALL-LABEL: and_i1_255:

  ; ALL:          move    $2, $4

  %r = and i1 255, %b
  ret i1 %r
}

define signext i8 @and_i8_255(i8 signext %b) {
entry:
; ALL-LABEL: and_i8_255:

  ; ALL:          move    $2, $4

  %r = and i8 255, %b
  ret i8 %r
}

define signext i16 @and_i16_255(i16 signext %b) {
entry:
; ALL-LABEL: and_i16_255:

  ; GP32:         andi    $2, $4, 255

  ; GP64:         andi    $2, $4, 255

  ; MM:           andi16  $2, $4, 255

  %r = and i16 255, %b
  ret i16 %r
}

define signext i32 @and_i32_255(i32 signext %b) {
entry:
; ALL-LABEL: and_i32_255:

  ; GP32:         andi    $2, $4, 255

  ; GP64:         andi    $2, $4, 255

  ; MM:           andi16  $2, $4, 255

  %r = and i32 255, %b
  ret i32 %r
}

define signext i64 @and_i64_255(i64 signext %b) {
entry:
; ALL-LABEL: and_i64_255:

  ; GP32:         andi    $3, $5, 255
  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         andi    $2, $4, 255

  ; MM32:         andi16  $3, $5, 255
  ; MM32:         li16    $2, 0

  ; MM64:         andi    $2, $4, 255

  %r = and i64 255, %b
  ret i64 %r
}

define signext i128 @and_i128_255(i128 signext %b) {
entry:
; ALL-LABEL: and_i128_255:

  ; GP32:         andi    $5, $7, 255
  ; GP32:         addiu   $2, $zero, 0
  ; GP32:         addiu   $3, $zero, 0
  ; GP32:         addiu   $4, $zero, 0

  ; GP64:         andi    $3, $5, 255
  ; GP64:         daddiu  $2, $zero, 0

  ; MM32:         andi16  $5, $7, 255
  ; MM32:         li16    $2, 0
  ; MM32:         li16    $3, 0
  ; MM32:         li16    $4, 0

  ; MM64:         andi    $3, $5, 255
  ; MM64:         daddiu  $2, $zero, 0

  %r = and i128 255, %b
  ret i128 %r
}

define signext i1 @and_i1_32768(i1 signext %b) {
entry:
; ALL-LABEL: and_i1_32768:

  ; GP32:         addiu  $2, $zero, 0

  ; GP64:         addiu  $2, $zero, 0

  ; MM:           li16   $2, 0

  %r = and i1 32768, %b
  ret i1 %r
}

define signext i8 @and_i8_32768(i8 signext %b) {
entry:
; ALL-LABEL: and_i8_32768:

  ; GP32:         addiu  $2, $zero, 0

  ; GP64:         addiu  $2, $zero, 0

  ; MM:           li16   $2, 0

  %r = and i8 32768, %b
  ret i8 %r
}

define signext i16 @and_i16_32768(i16 signext %b) {
entry:
; ALL-LABEL: and_i16_32768:

  ; GP32:         addiu  $[[T0:[0-9]+]], $zero, -32768
  ; GP32:         and    $2, $4, $[[T0]]

  ; GP64:         addiu  $[[T0:[0-9]+]], $zero, -32768
  ; GP64:         and    $2, $4, $[[T0]]

  ; MM:           addiu  $2, $zero, -32768
  ; MM:           and16  $2, $4

  %r = and i16 32768, %b
  ret i16 %r
}

define signext i32 @and_i32_32768(i32 signext %b) {
entry:
; ALL-LABEL: and_i32_32768:

  ; GP32:         andi    $2, $4, 32768

  ; GP64:         andi    $2, $4, 32768

  ; MM:           andi16  $2, $4, 32768

  %r = and i32 32768, %b
  ret i32 %r
}

define signext i64 @and_i64_32768(i64 signext %b) {
entry:
; ALL-LABEL: and_i64_32768:

  ; GP32:         andi    $3, $5, 32768
  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         andi    $2, $4, 32768

  ; MM32:         andi16  $3, $5, 32768
  ; MM32:         li16    $2, 0

  ; MM64:         andi    $2, $4, 32768

  %r = and i64 32768, %b
  ret i64 %r
}

define signext i128 @and_i128_32768(i128 signext %b) {
entry:
; ALL-LABEL: and_i128_32768:

  ; GP32:         andi    $5, $7, 32768
  ; GP32:         addiu   $2, $zero, 0
  ; GP32:         addiu   $3, $zero, 0
  ; GP32:         addiu   $4, $zero, 0

  ; GP64:         andi    $3, $5, 32768
  ; GP64:         daddiu  $2, $zero, 0

  ; MM32:         andi16  $5, $7, 32768
  ; MM32:         li16    $2, 0
  ; MM32:         li16    $3, 0
  ; MM32:         li16    $4, 0

  ; MM64:         andi    $3, $5, 32768
  ; MM64:         daddiu  $2, $zero, 0

  %r = and i128 32768, %b
  ret i128 %r
}

define signext i1 @and_i1_65(i1 signext %b) {
entry:
; ALL-LABEL: and_i1_65:

  ; ALL:          move    $2, $4

  %r = and i1 65, %b
  ret i1 %r
}

define signext i8 @and_i8_65(i8 signext %b) {
entry:
; ALL-LABEL: and_i8_65:

  ; ALL:          andi    $2, $4, 65

  %r = and i8 65, %b
  ret i8 %r
}

define signext i16 @and_i16_65(i16 signext %b) {
entry:
; ALL-LABEL: and_i16_65:

  ; ALL:          andi    $2, $4, 65

  %r = and i16 65, %b
  ret i16 %r
}

define signext i32 @and_i32_65(i32 signext %b) {
entry:
; ALL-LABEL: and_i32_65:

  ; ALL:          andi    $2, $4, 65

  %r = and i32 65, %b
  ret i32 %r
}

define signext i64 @and_i64_65(i64 signext %b) {
entry:
; ALL-LABEL: and_i64_65:

  ; GP32:         andi    $3, $5, 65
  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         andi    $2, $4, 65

  ; MM32-DAG:     andi    $3, $5, 65
  ; MM32-DAG:     li16    $2, 0

  ; MM64:         andi    $2, $4, 65

  %r = and i64 65, %b
  ret i64 %r
}

define signext i128 @and_i128_65(i128 signext %b) {
entry:
; ALL-LABEL: and_i128_65:

  ; GP32:         andi    $5, $7, 65
  ; GP32:         addiu   $2, $zero, 0
  ; GP32:         addiu   $3, $zero, 0
  ; GP32:         addiu   $4, $zero, 0

  ; GP64:         andi    $3, $5, 65
  ; GP64:         daddiu  $2, $zero, 0

  ; MM32-DAG:     andi    $5, $7, 65
  ; MM32-DAG:     li16    $2, 0
  ; MM32-DAG:     li16    $3, 0
  ; MM32-DAG:     li16    $4, 0

  ; MM64:         andi    $3, $5, 65
  ; MM64:         daddiu  $2, $zero, 0

  %r = and i128 65, %b
  ret i128 %r
}

define signext i1 @and_i1_256(i1 signext %b) {
entry:
; ALL-LABEL: and_i1_256:

  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         addiu   $2, $zero, 0

  ; MM:           li16    $2, 0

  %r = and i1 256, %b
  ret i1 %r
}

define signext i8 @and_i8_256(i8 signext %b) {
entry:
; ALL-LABEL: and_i8_256:

  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         addiu   $2, $zero, 0

  ; MM:           li16    $2, 0

  %r = and i8 256, %b
  ret i8 %r
}

define signext i16 @and_i16_256(i16 signext %b) {
entry:
; ALL-LABEL: and_i16_256:

  ; ALL:          andi    $2, $4, 256

  %r = and i16 256, %b
  ret i16 %r
}

define signext i32 @and_i32_256(i32 signext %b) {
entry:
; ALL-LABEL: and_i32_256:

  ; ALL:          andi    $2, $4, 256

  %r = and i32 256, %b
  ret i32 %r
}

define signext i64 @and_i64_256(i64 signext %b) {
entry:
; ALL-LABEL: and_i64_256:

  ; GP32:         andi    $3, $5, 256
  ; GP32:         addiu   $2, $zero, 0

  ; GP64:         andi    $2, $4, 256

  ; MM32-DAG:     andi    $3, $5, 256
  ; MM32-DAG:     li16    $2, 0

  ; MM64:         andi    $2, $4, 256

  %r = and i64 256, %b
  ret i64 %r
}

define signext i128 @and_i128_256(i128 signext %b) {
entry:
; ALL-LABEL: and_i128_256:

  ; GP32:         andi    $5, $7, 256
  ; GP32:         addiu   $2, $zero, 0
  ; GP32:         addiu   $3, $zero, 0
  ; GP32:         addiu   $4, $zero, 0

  ; GP64:         andi    $3, $5, 256
  ; GP64:         daddiu  $2, $zero, 0

  ; MM32-DAG:     andi    $5, $7, 256
  ; MM32-DAG:     li16    $2, 0
  ; MM32-DAG:     li16    $3, 0
  ; MM32-DAG:     li16    $4, 0

  ; MM64:         andi    $3, $5, 256
  ; MM64:         daddiu  $2, $zero, 0

  %r = and i128 256, %b
  ret i128 %r
}
