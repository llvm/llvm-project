; Test linking two functions with different prototypes and two globals 
; in different modules.
; RUN: llvm-as %s -o %t.foo1.bc
; RUN: echo | llvm-as -o %t.foo2.bc
; RUN: llvm-link %t.foo2.bc %t.foo1.bc -S | grep "addrspace(2)"
; RUN: llvm-link %t.foo1.bc %t.foo2.bc -S | grep "addrspace(2)"
; rdar://6038021

@G = addrspace(2) global i32 256 
