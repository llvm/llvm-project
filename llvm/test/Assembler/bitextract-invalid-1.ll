; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s
; CHECK: invalid bitextract operands
define i8 @invalid(i32 %src) {
  %r = bitextract i8, i32 %src, i32 0
  ret i8 %r
}
