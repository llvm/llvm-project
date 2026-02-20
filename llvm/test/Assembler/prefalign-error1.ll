; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: expected '('
define void @error() prefalign {
  ret void
}
