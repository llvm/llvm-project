; RUN: llc < %s -mtriple=mips64el-unknown-linux-gnu -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefix=MIPS4
; RUN: llc < %s -mtriple=mips64el-unknown-linux-gnu -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefix=MIPS64R2

define i64 @foo(i64 noundef %a) {
; MIPS4-LABEL: foo:
; MIPS4:       # %bb.0: # %entry
; MIPS4-NEXT:   sll    $1, $4, 0
; MIPS4-NEXT:   srl    $1, $1, 2
; MIPS4-NEXT:   andi   $1, $1, 7
; MIPS4-NEXT:   daddiu $2, $zero, 1
; MIPS4-NEXT:   jr     $ra
; MIPS4-NEXT:   dsllv  $2, $2, $1
;
; MIPS64R2-LABEL: foo:
; MIPS64R2:       # %bb.0: # %entry
; MIPS64R2-NEXT:   sll	  $1, $4, 0
; MIPS64R2-NEXT:   ext	  $1, $1, 2, 3
; MIPS64R2-NEXT:   daddiu $2, $zero, 1
; MIPS64R2-NEXT:   jr     $ra
; MIPS64R2-NEXT:   dsllv  $2, $2, $1
entry:
  %div1 = lshr i64 %a, 2
  %and = and i64 %div1, 7
  %shl = shl nuw nsw i64 1, %and
  ret i64 %shl
}
