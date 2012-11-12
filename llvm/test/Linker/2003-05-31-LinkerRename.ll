; The funcresolve pass will (intentionally) llvm-link an _internal_ function 
; body with an external declaration.  Because of this, if we LINK an internal 
; function body into a program that already has an external declaration for 
; the function name, we must rename the internal function to something that 
; does not conflict.

; RUN: echo " define internal i32 @foo() { ret i32 7 } " | llvm-as > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | grep internal | not grep "@foo("

declare i32 @foo() 

define i32 @test() { 
  %X = call i32 @foo()
  ret i32 %X
}

