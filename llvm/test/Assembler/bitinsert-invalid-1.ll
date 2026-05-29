; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s
; CHECK: invalid bitinsert operands
define i32 @invalid(i32 %base, i8 %val) {
  %r = bitinsert i32 %base, i8 %val, i32 0
  ret i32 %r
}
