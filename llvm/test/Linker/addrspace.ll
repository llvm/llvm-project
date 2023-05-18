; RUN: llvm-link %s -S | FileCheck %s

@G = addrspace(2) global i32 256 
; CHECK: @G = addrspace(2) global i32

@GA = alias i32, ptr addrspace(2) @G
; CHECK: @GA = alias i32, ptr addrspace(2) @G

define void @foo() addrspace(3) {
; CHECK: define void @foo() addrspace(3)
  ret void
}
