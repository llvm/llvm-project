; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s
; PR1577

; CHECK: expected top-level entity

@anInt = global i32 1
alias i32 @anAlias

define i32 @main() {
   ret i32 0
}
