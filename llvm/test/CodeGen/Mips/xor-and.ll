; RUN: llc -O3 -mcpu=mips64 -mtriple=mips64el-unknown-linux-gnuabi64 < %s -o - | FileCheck %s

define noundef signext i32 @xor_and(i32 noundef signext %a, i32 noundef signext %b) local_unnamed_addr {
; CHECK-LABEL: xor_and:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    and $1, $5, $4
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    xor $2, $1, 4294967295

entry:
  %0 = and i32 %b, %a
  %or1 = xor i32 %0, -1
  ret i32 %or1
}
