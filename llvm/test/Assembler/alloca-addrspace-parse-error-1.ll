; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A1"

; addrspace and align in wrong order
; CHECK: :8:39: error: expected metadata after comma
define void @use_alloca() {
  %alloca = alloca i32, addrspace(1), align 4
  ret void
}

!0 = !{}
