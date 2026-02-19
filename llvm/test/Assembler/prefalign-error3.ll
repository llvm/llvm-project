; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: alignment is not a power of two
define void @error() prefalign(0) {
  ret void
}
