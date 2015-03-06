; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=NOT-R2-R6 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=NOT-R2-R6 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=R2-R6 -check-prefix=GP32
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=R2-R6 -check-prefix=GP32
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=NOT-R2-R6 -check-prefix=GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=NOT-R2-R6 -check-prefix=GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=R2-R6 -check-prefix=GP64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefix=ALL -check-prefix=R2-R6 -check-prefix=GP64

define signext i1 @add_i1(i1 signext %a, i1 signext %b) {
entry:
; ALL-LABEL: add_i1:

  ; ALL:        addu    $[[T0:[0-9]+]], $4, $5
  ; ALL:        sll     $[[T0]], $[[T0]], 31
  ; ALL:        sra     $2, $[[T0]], 31

  %r = add i1 %a, %b
  ret i1 %r
}

define signext i8 @add_i8(i8 signext %a, i8 signext %b) {
entry:
; ALL-LABEL: add_i8:

  ; NOT-R2-R6:  addu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:  sll     $[[T0]], $[[T0]], 24
  ; NOT-R2-R6:  sra     $2, $[[T0]], 24

  ; R2-R6:         addu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:         seb     $2, $[[T0:[0-9]+]]

  %r = add i8 %a, %b
  ret i8 %r
}

define signext i16 @add_i16(i16 signext %a, i16 signext %b) {
entry:
; ALL-LABEL: add_i16:

  ; NOT-R2-R6:  addu    $[[T0:[0-9]+]], $4, $5
  ; NOT-R2-R6:  sll     $[[T0]], $[[T0]], 16
  ; NOT-R2-R6:  sra     $2, $[[T0]], 16

  ; R2-R6:         addu    $[[T0:[0-9]+]], $4, $5
  ; R2-R6:         seh     $2, $[[T0:[0-9]+]]

  %r = add i16 %a, %b
  ret i16 %r
}

define signext i32 @add_i32(i32 signext %a, i32 signext %b) {
entry:
; ALL-LABEL: add_i32:

  ; ALL:        addu    $2, $4, $5

  %r = add i32 %a, %b
  ret i32 %r
}

define signext i64 @add_i64(i64 signext %a, i64 signext %b) {
entry:
; ALL-LABEL: add_i64:

  ; GP32:       addu    $3, $5, $7
  ; GP32:       sltu    $[[T0:[0-9]+]], $3, $7
  ; GP32:       addu    $[[T1:[0-9]+]], $[[T0]], $6
  ; GP32:       addu    $2, $4, $[[T1]]

  ; GP64:       daddu   $2, $4, $5

  %r = add i64 %a, %b
  ret i64 %r
}

define signext i128 @add_i128(i128 signext %a, i128 signext %b) {
entry:
; ALL-LABEL: add_i128:

  ; GP32:       lw        $[[T0:[0-9]+]], 28($sp)
  ; GP32:       addu      $[[T1:[0-9]+]], $7, $[[T0]]
  ; GP32:       sltu      $[[T2:[0-9]+]], $[[T1]], $[[T0]]
  ; GP32:       lw        $[[T3:[0-9]+]], 24($sp)
  ; GP32:       addu      $[[T4:[0-9]+]], $[[T2]], $[[T3]]
  ; GP32:       addu      $[[T5:[0-9]+]], $6, $[[T4]]
  ; GP32:       sltu      $[[T6:[0-9]+]], $[[T5]], $[[T3]]
  ; GP32:       lw        $[[T7:[0-9]+]], 20($sp)
  ; GP32:       addu      $[[T8:[0-9]+]], $[[T6]], $[[T7]]
  ; GP32:       lw        $[[T9:[0-9]+]], 16($sp)
  ; GP32:       addu      $3, $5, $[[T8]]
  ; GP32:       sltu      $[[T10:[0-9]+]], $3, $[[T7]]
  ; GP32:       addu      $[[T11:[0-9]+]], $[[T10]], $[[T9]]
  ; GP32:       addu      $2, $4, $[[T11]]
  ; GP32:       move      $4, $[[T5]]
  ; GP32:       move      $5, $[[T1]]

  ; GP64:       daddu     $3, $5, $7
  ; GP64:       sltu      $[[T0:[0-9]+]], $3, $7
  ; GP64:       daddu     $[[T1:[0-9]+]], $[[T0]], $6
  ; GP64:       daddu     $2, $4, $[[T1]]

  %r = add i128 %a, %b
  ret i128 %r
}
