; RUN: not llvm-as < %s > /dev/null |& grep {expected top-level entity}
; PR1577

@anInt = global i32 1 
alias i32 @anAlias

define i32 @main() {
   ret i32 0 
}
