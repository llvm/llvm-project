; RUN: opt -passes=instsimplify -S %s | FileCheck %s

declare i32 @llvm.uadd.sat.i32(i32, i32)
declare i32 @llvm.usub.sat.i32(i32, i32)

; CHECK-LABEL: define i1 @add_ge_or
; CHECK-NEXT: ret i1 true
define i1 @add_ge_or(i32 %x, i32 %y) {
  %sat = call i32 @llvm.uadd.sat.i32(i32 %x, i32 %y)
  %v = or i32 %x, %y
  %cmp = icmp uge i32 %sat, %v
  ret i1 %cmp
}

define i1 @sub_le_andnot(i32 %x, i32 %y) {
  %sat = call i32 @llvm.usub.sat.i32(i32 %x, i32 %y)
  %noty = xor i32 %y, -1
  %v = and i32 %x, %noty
  %cmp = icmp ule i32 %sat, %v
  ret i1 %cmp
}

define i1 @sub_le_not(i32 %x, i32 %y) {
  %sat = call i32 @llvm.usub.sat.i32(i32 %x, i32 %y)
  %noty = xor i32 %y, -1
  %cmp = icmp ule i32 %sat, %noty
  ret i1 %cmp
}

