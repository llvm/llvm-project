; This fails because the linker renames the external symbol not the internal 
; one...

; RUN: echo "define internal i32 @foo() { ret i32 7 } " | llvm-as > %t.1.bc
; RUN: llvm-as %s -o %t.2.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | grep "@foo()" | grep -v internal

define i32 @foo() { ret i32 0 }

