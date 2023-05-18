; Test return attributes
; RUN: llvm-as < %s | llvm-dis | grep "define inreg i32"
; RUN: llvm-as < %s | llvm-dis | grep "call inreg i32"
; RUN: verify-uselistorder %s

define inreg i32 @fn1() {
  ret i32 0
}

define void @fn2() {
  %t = call inreg i32 @fn1()
  ret void
}

