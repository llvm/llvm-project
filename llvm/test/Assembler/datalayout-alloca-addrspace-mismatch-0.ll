; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A1"

; CHECK: :7:41: error: address space must match datalayout
define void @use_alloca() {
  %alloca_scalar_no_align = alloca i32, addrspace(2)
  ret void
}
