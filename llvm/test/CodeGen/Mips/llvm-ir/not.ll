; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips2 | FileCheck %s -check-prefixes=GP32
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32 | FileCheck %s -check-prefixes=GP32
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=GP32
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=GP32
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=GP32
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=GP32
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips64-unknown-linux-gnu -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=GP64
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32r3 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=MM,MM32r3
; RUN: llc < %s -mtriple=mips-unknown-linux-gnu -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=MM,MM32r6

define signext i1 @not_i1(i1 signext %a) {
; GP32-LABEL: not_i1:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    not $2, $4
;
; GP64-LABEL: not_i1:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    jr $ra
; GP64-NEXT:    not $2, $4
;
; MM-LABEL: not_i1:
; MM:       # %bb.0: # %entry
; MM-NEXT:    not16 $2, $4
; MM-NEXT:    jrc $ra
entry:
  %r = xor i1 %a, -1
  ret i1 %r
}

define signext i8 @not_i8(i8 signext %a) {
; GP32-LABEL: not_i8:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    not $2, $4
;
; GP64-LABEL: not_i8:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    jr $ra
; GP64-NEXT:    not $2, $4
;
; MM-LABEL: not_i8:
; MM:       # %bb.0: # %entry
; MM-NEXT:    not16 $2, $4
; MM-NEXT:    jrc $ra
entry:
  %r = xor i8 %a, -1
  ret i8 %r
}

define signext i16 @not_i16(i16 signext %a) {
; GP32-LABEL: not_i16:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    not $2, $4
;
; GP64-LABEL: not_i16:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    jr $ra
; GP64-NEXT:    not $2, $4
;
; MM-LABEL: not_i16:
; MM:       # %bb.0: # %entry
; MM-NEXT:    not16 $2, $4
; MM-NEXT:    jrc $ra
entry:
  %r = xor i16 %a, -1
  ret i16 %r
}

define signext i32 @not_i32(i32 signext %a) {
; GP32-LABEL: not_i32:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    not $2, $4
;
; GP64-LABEL: not_i32:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    daddiu $1, $zero, -1
; GP64-NEXT:    jr $ra
; GP64-NEXT:    xor $2, $4, $1
;
; MM-LABEL: not_i32:
; MM:       # %bb.0: # %entry
; MM-NEXT:    not16 $2, $4
; MM-NEXT:    jrc $ra
entry:
  %r = xor i32 %a, -1
  ret i32 %r
}

define signext i64 @not_i64(i64 signext %a) {
; GP32-LABEL: not_i64:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    not $2, $4
; GP32-NEXT:    jr $ra
; GP32-NEXT:    not $3, $5
;
; GP64-LABEL: not_i64:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    daddiu $1, $zero, -1
; GP64-NEXT:    jr $ra
; GP64-NEXT:    xor $2, $4, $1
;
; MM-LABEL: not_i64:
; MM:       # %bb.0: # %entry
; MM-NEXT:    not16 $2, $4
; MM-NEXT:    not16 $3, $5
; MM-NEXT:    jrc $ra
entry:
  %r = xor i64 %a, -1
  ret i64 %r
}

define signext i128 @not_i128(i128 signext %a) {
; GP32-LABEL: not_i128:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    not $2, $4
; GP32-NEXT:    not $3, $5
; GP32-NEXT:    not $4, $6
; GP32-NEXT:    jr $ra
; GP32-NEXT:    not $5, $7
;
; GP64-LABEL: not_i128:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    daddiu $1, $zero, -1
; GP64-NEXT:    xor $2, $4, $1
; GP64-NEXT:    jr $ra
; GP64-NEXT:    xor $3, $5, $1
;
; MM-LABEL: not_i128:
; MM:       # %bb.0: # %entry
; MM-NEXT:    not16 $2, $4
; MM-NEXT:    not16 $3, $5
; MM-NEXT:    not16 $4, $6
; MM-NEXT:    not16 $5, $7
; MM-NEXT:    jrc $ra
entry:
  %r = xor i128 %a, -1
  ret i128 %r
}

define signext i1 @nor_i1(i1 signext %a, i1 signext %b) {
; GP32-LABEL: nor_i1:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    nor $2, $5, $4
;
; GP64-LABEL: nor_i1:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    or $1, $5, $4
; GP64-NEXT:    sll $1, $1, 0
; GP64-NEXT:    jr $ra
; GP64-NEXT:    not $2, $1
;
; MM32r3-LABEL: nor_i1:
; MM32r3:       # %bb.0: # %entry
; MM32r3-NEXT:    jr $ra
; MM32r3-NEXT:    nor $2, $5, $4
;
; MM32r6-LABEL: nor_i1:
; MM32r6:       # %bb.0: # %entry
; MM32r6-NEXT:    nor $2, $5, $4
; MM32r6-NEXT:    jrc $ra
entry:
  %or = or i1 %b, %a
  %r = xor i1 %or, -1
  ret i1 %r
}

define signext i8 @nor_i8(i8 signext %a, i8 signext %b) {
; GP32-LABEL: nor_i8:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    nor $2, $5, $4
;
; GP64-LABEL: nor_i8:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    or $1, $5, $4
; GP64-NEXT:    sll $1, $1, 0
; GP64-NEXT:    jr $ra
; GP64-NEXT:    not $2, $1
;
; MM32r3-LABEL: nor_i8:
; MM32r3:       # %bb.0: # %entry
; MM32r3-NEXT:    jr $ra
; MM32r3-NEXT:    nor $2, $5, $4
;
; MM32r6-LABEL: nor_i8:
; MM32r6:       # %bb.0: # %entry
; MM32r6-NEXT:    nor $2, $5, $4
; MM32r6-NEXT:    jrc $ra
entry:
  %or = or i8 %b, %a
  %r = xor i8 %or, -1
  ret i8 %r
}

define signext i16 @nor_i16(i16 signext %a, i16 signext %b) {
; GP32-LABEL: nor_i16:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    nor $2, $5, $4
;
; GP64-LABEL: nor_i16:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    or $1, $5, $4
; GP64-NEXT:    sll $1, $1, 0
; GP64-NEXT:    jr $ra
; GP64-NEXT:    not $2, $1
;
; MM32r3-LABEL: nor_i16:
; MM32r3:       # %bb.0: # %entry
; MM32r3-NEXT:    jr $ra
; MM32r3-NEXT:    nor $2, $5, $4
;
; MM32r6-LABEL: nor_i16:
; MM32r6:       # %bb.0: # %entry
; MM32r6-NEXT:    nor $2, $5, $4
; MM32r6-NEXT:    jrc $ra
entry:
  %or = or i16 %b, %a
  %r = xor i16 %or, -1
  ret i16 %r
}

define signext i32 @nor_i32(i32 signext %a, i32 signext %b) {
; GP32-LABEL: nor_i32:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    jr $ra
; GP32-NEXT:    nor $2, $5, $4
;
; GP64-LABEL: nor_i32:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    jr $ra
; GP64-NEXT:    nor $2, $5, $4
;
; MM32r3-LABEL: nor_i32:
; MM32r3:       # %bb.0: # %entry
; MM32r3-NEXT:    jr $ra
; MM32r3-NEXT:    nor $2, $5, $4
;
; MM32r6-LABEL: nor_i32:
; MM32r6:       # %bb.0: # %entry
; MM32r6-NEXT:    nor $2, $5, $4
; MM32r6-NEXT:    jrc $ra
entry:
  %or = or i32 %b, %a
  %r = xor i32 %or, -1
  ret i32 %r
}

define signext i64 @nor_i64(i64 signext %a, i64 signext %b) {
; GP32-LABEL: nor_i64:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    nor $2, $6, $4
; GP32-NEXT:    jr $ra
; GP32-NEXT:    nor $3, $7, $5
;
; GP64-LABEL: nor_i64:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    jr $ra
; GP64-NEXT:    nor $2, $5, $4
;
; MM32r3-LABEL: nor_i64:
; MM32r3:       # %bb.0: # %entry
; MM32r3-NEXT:    nor $2, $6, $4
; MM32r3-NEXT:    jr $ra
; MM32r3-NEXT:    nor $3, $7, $5
;
; MM32r6-LABEL: nor_i64:
; MM32r6:       # %bb.0: # %entry
; MM32r6-NEXT:    nor $2, $6, $4
; MM32r6-NEXT:    nor $3, $7, $5
; MM32r6-NEXT:    jrc $ra
entry:
  %or = or i64 %b, %a
  %r = xor i64 %or, -1
  ret i64 %r
}

define signext i128 @nor_i128(i128 signext %a, i128 signext %b) {
; GP32-LABEL: nor_i128:
; GP32:       # %bb.0: # %entry
; GP32-NEXT:    lw $1, 20($sp)
; GP32-NEXT:    lw $2, 16($sp)
; GP32-NEXT:    nor $2, $2, $4
; GP32-NEXT:    nor $3, $1, $5
; GP32-NEXT:    lw $1, 24($sp)
; GP32-NEXT:    nor $4, $1, $6
; GP32-NEXT:    lw $1, 28($sp)
; GP32-NEXT:    jr $ra
; GP32-NEXT:    nor $5, $1, $7
;
; GP64-LABEL: nor_i128:
; GP64:       # %bb.0: # %entry
; GP64-NEXT:    nor $2, $6, $4
; GP64-NEXT:    jr $ra
; GP64-NEXT:    nor $3, $7, $5
;
; MM32r3-LABEL: nor_i128:
; MM32r3:       # %bb.0: # %entry
; MM32r3-NEXT:    lw $1, 20($sp)
; MM32r3-NEXT:    lw $2, 16($sp)
; MM32r3-NEXT:    nor $2, $2, $4
; MM32r3-NEXT:    nor $3, $1, $5
; MM32r3-NEXT:    lw $1, 24($sp)
; MM32r3-NEXT:    nor $4, $1, $6
; MM32r3-NEXT:    lw $1, 28($sp)
; MM32r3-NEXT:    jr $ra
; MM32r3-NEXT:    nor $5, $1, $7
;
; MM32r6-LABEL: nor_i128:
; MM32r6:       # %bb.0: # %entry
; MM32r6-NEXT:    lw $1, 20($sp)
; MM32r6-NEXT:    lw $2, 16($sp)
; MM32r6-NEXT:    nor $2, $2, $4
; MM32r6-NEXT:    nor $3, $1, $5
; MM32r6-NEXT:    lw $1, 24($sp)
; MM32r6-NEXT:    nor $4, $1, $6
; MM32r6-NEXT:    lw $1, 28($sp)
; MM32r6-NEXT:    nor $5, $1, $7
; MM32r6-NEXT:    jrc $ra
entry:
  %or = or i128 %b, %a
  %r = xor i128 %or, -1
  ret i128 %r
}
