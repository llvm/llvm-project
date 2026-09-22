; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare pisa_kernel void @kernel()

define void @caller() {
  call pisa_kernel void @kernel()
  ret void
}

; CHECK: calling convention does not permit calls
