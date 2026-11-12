; Test linking two functions with different prototypes and two globals 
; in different modules.
; RUN: llvm-as %s -o %t.foo1.bc
; RUN: llvm-as %s -o %t.foo2.bc
; RUN: echo "define linkonce void @foo(i32 %x) { ret void }" | llvm-as -o %t.foo3.bc
; RUN: llvm-link %t.foo1.bc %t.foo2.bc -S
; RUN: llvm-link %t.foo1.bc %t.foo3.bc -S
define linkonce void @foo() { ret void }
