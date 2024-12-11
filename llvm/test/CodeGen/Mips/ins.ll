; RUN: llc -O3 -mcpu=mips32r2 -mtriple=mipsel-linux-gnu < %s -o - \
; RUN:   | FileCheck %s --check-prefixes=MIPS32R2
; RUN: llc -O3 -mcpu=mips64r2 -mtriple=mips64el  < %s \
; RUN:   | FileCheck %s --check-prefixes=MIPS64R2

define i32 @or_and_shl(i32 %a, i32 %b) {
; MIPS32R2-LABEL: or_and_shl:
; MIPS32R2:       # %bb.0: # %entry
; MIPS32R2-NEXT:    ins $4, $5, 31, 1
; MIPS32R2-NEXT:    jr $ra
; MIPS32R2-NEXT:    move $2, $4

entry:
  %shl = shl i32 %b, 31
  %and = and i32 %a, 2147483647
  %or = or i32 %and, %shl
  ret i32 %or
}

define i32 @or_shl_and(i32 %a, i32 %b) {
; MIPS32R2-LABEL: or_shl_and:
; MIPS32R2:       # %bb.0: # %entry
; MIPS32R2-NEXT:    ins $4, $5, 31, 1
; MIPS32R2-NEXT:    jr $ra
; MIPS32R2-NEXT:    move $2, $4

entry:
  %shl = shl i32 %b, 31
  %and = and i32 %a, 2147483647
  %or = or i32 %shl, %and
  ret i32 %or
}

define i64 @dinsm(i64 %a, i64 %b) {
; MIPS64R2-LABEL: dinsm:
; MIPS64R2:       # %bb.0: # %entry
; MIPS64R2-NEXT:    dinsm $4, $5, 17, 47
; MIPS64R2-NEXT:    jr $ra
; MIPS64R2-NEXT:    move $2, $4

entry:
  %shl = shl i64 %b, 17
  %and = and i64 %a, 131071
  %or = or i64 %shl, %and
  ret i64 %or
}

define i64 @dinsu(i64 %a, i64 %b) {
; MIPS64R2-LABEL: dinsu:
; MIPS64R2:       # %bb.0: # %entry
; MIPS64R2-NEXT:    dinsu $4, $5, 35, 29
; MIPS64R2-NEXT:    jr $ra
; MIPS64R2-NEXT:    move $2, $4

entry:
  %shl = shl i64 %b, 35
  %and = and i64 %a, 34359738367
  %or = or i64 %shl, %and
  ret i64 %or
}
