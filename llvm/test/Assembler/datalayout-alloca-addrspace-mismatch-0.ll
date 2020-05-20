; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A1"

; CHECK: Allocation instruction pointer not in the stack address space!
; CHECK-NEXT:  %alloca_scalar_no_align = alloca i32, align 4, addrspace(2)

define void @use_alloca() {
  %alloca_scalar_no_align = alloca i32, addrspace(2)
  ret void
}
