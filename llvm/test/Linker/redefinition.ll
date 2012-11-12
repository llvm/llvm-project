; Test linking two functions with different prototypes and two globals 
; in different modules.
; RUN: llvm-as %s -o %t.foo1.bc
; RUN: llvm-as %s -o %t.foo2.bc
; RUN: echo "define void @foo(i32 %x) { ret void }" | llvm-as -o %t.foo3.bc
; RUN: not llvm-link %t.foo1.bc %t.foo2.bc -o %t.bc 2>&1 | \
; RUN:   grep "symbol multiply defined"
; RUN: not llvm-link %t.foo1.bc %t.foo3.bc -o %t.bc 2>&1 | \
; RUN:   grep "symbol multiply defined"
define void @foo() { ret void }
