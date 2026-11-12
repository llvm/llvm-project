; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A1"

; CHECK: :8:3: error: expected metadata after comma
define void @use_alloca() {
  %alloca = alloca i32, addrspace(1),
  ret void
}

!0 = !{}
