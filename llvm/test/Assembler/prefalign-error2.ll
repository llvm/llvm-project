; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: expected integer
define void @error() prefalign() {
  ret void
}
