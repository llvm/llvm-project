; RUN: opt -passes=instcombine -S %s | FileCheck %s

define i64 @rotate_not(i64 %x, i64 %y) {
; CHECK: call i64 @llvm.fshl.i64
  %ny = xor i64 %y, -1
  %a = lshr i64 %x, %ny
  %b = add i64 %y, 1
  %c = shl i64 %x, %b
  %d = or i64 %a, %c
  ret i64 %d
}
