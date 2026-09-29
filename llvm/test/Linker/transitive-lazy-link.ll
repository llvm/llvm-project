; @f and @g are lazily linked. @f requires @g - ensure @g is correctly linked.

; RUN: echo "declare i32 @f(i32)" > %t.1.ll
; RUN: echo "define i32 @h(i32 %x) {" >> %t.1.ll
; RUN: echo "  %1 = call i32 @f(i32 %x)" >> %t.1.ll
; RUN: echo "  ret i32 %1" >> %t.1.ll
; RUN: echo "}" >> %t.1.ll
; RUN: llvm-as < %t.1.ll > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; RUN: llvm-link %t.1.bc %t.2.bc

define available_externally i32 @f(i32 %x) {
  %1 = call i32 @g(i32 %x)
  ret i32 %1
}

define available_externally i32 @g(i32 %x) {
  ret i32 5
}

