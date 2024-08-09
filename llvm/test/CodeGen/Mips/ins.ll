; RUN: llc -O3 -mcpu=mips64r2 -mtriple=mips64el-unknown-linux-gnuabi64 < %s -o - | FileCheck %s

define void @or_and_shl(ptr nocapture noundef %a, i64 noundef signext %b) {
; CHECK-LABEL: or_and_shl:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lw $1, 0($4)
; CHECK-NEXT:    sll $2, $5, 0
; CHECK-NEXT:    ins $1, $2, 31, 1
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    sw $1, 0($4)

entry:
  %conv = trunc i64 %b to i32
  %load = load i32, ptr %a, align 4
  %shl = shl i32 %conv, 31
  %and = and i32 %load, 2147483647
  %or = or i32 %and, %shl
  store i32 %or, ptr %a, align 4
  ret void
}


