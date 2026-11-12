; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Check that parser rejects address spaces that are too large to fit in the 24 bits

define void @f() {
; CHECK: invalid address space, must be a 24-bit integer
  %y = alloca i32, addrspace(16777216)
  ret void
}
