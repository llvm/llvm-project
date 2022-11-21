; RUN: not --crash llc -mtriple=riscv64 -mattr=+f,+d -verify-machineinstrs -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=riscv32 -mattr=+f,+d -verify-machineinstrs -filetype=null < %s 2>&1 | FileCheck %s

define ghccc ptr @nest_receiver(ptr nest %arg) nounwind {
  ret ptr %arg
}

define ghccc ptr @nest_caller(ptr %arg) nounwind {
  %result = call ghccc ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}

; CHECK: LLVM ERROR: Attribute 'nest' is not supported in GHC calling convention
