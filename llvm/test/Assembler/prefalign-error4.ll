; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: huge alignments are not supported yet
define void @error() prefalign(8589934592) {
  ret void
}
