; Test linking two functions with different prototypes and two globals 
; in different modules.
; RUN: llvm-as %s -o %t.foo1.bc -f
; RUN: llvm-as %s -o %t.foo2.bc -f
; RUN: echo {define linkonce void @foo(i32 %x) { ret void }} | llvm-as -o %t.foo3.bc -f
; RUN: llvm-link %t.foo1.bc %t.foo2.bc | llvm-dis
; RUN: llvm-link %t.foo1.bc %t.foo3.bc | llvm-dis
define linkonce void @foo() { ret void }
