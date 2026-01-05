; RUN: llc -O3 -mcpu=mips64 -mtriple=mips64el-unknown-linux-gnuabi64 < %s -o - | FileCheck %s

define noundef signext i32 @xor_and(i32 noundef signext %a, i32 noundef signext %b) local_unnamed_addr {
; CHECK-LABEL: xor_and:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    and $1, $5, $4
; CHECK-NEXT:    daddiu $2, $zero, -1
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    xor $2, $1, $2

entry:
  %0 = and i32 %b, %a
  %or1 = xor i32 %0, -1
  ret i32 %or1
}

define noundef signext i32 @input_i16(i16 noundef signext %a, i16 noundef signext %b) local_unnamed_addr {
; CHECK-LABEL: input_i16:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    and $1, $5, $4
; CHECK-NEXT:    daddiu $2, $zero, -1
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    xor $2, $1, $2

entry:
  %0 = and i16 %b, %a
  %1 = xor i16 %0, -1
  %or4 = sext i16 %1 to i32
  ret i32 %or4
}

define i64 @return_i64(i32 noundef signext %a, i32 noundef signext %b) local_unnamed_addr {
; CHECK-LABEL: return_i64:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    and $1, $5, $4
; CHECK-NEXT:    daddiu $2, $zero, -1
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    xor $2, $1, $2

entry:
  %0 = and i32 %b, %a
  %or1 = xor i32 %0, -1
  %conv = sext i32 %or1 to i64
  ret i64 %conv
}
