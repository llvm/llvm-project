; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: comdat cannot be unnamed

define void @0() comdat {
  ret void
}
