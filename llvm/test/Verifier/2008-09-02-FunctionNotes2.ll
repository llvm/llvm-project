; Test function notes
; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attributes 'noinline and alwaysinline' are incompatible
define void @fn1() alwaysinline noinline {
  ret void
}
