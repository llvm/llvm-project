; RUN: not llvm-as < %s > /dev/null 2> %t
; RUN: grep "expected top-level entity" %t
; PR1577

@anInt = global i32 1 
alias i32 @anAlias

define i32 @main() {
   ret i32 0 
}
