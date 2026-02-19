; RUN: not llvm-as %s 2>&1 | FileCheck %s

define void @error() prefalign {
  ; CHECK: expected '('
  ret void
}
