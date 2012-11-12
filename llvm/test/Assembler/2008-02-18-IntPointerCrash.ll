; RUN: not llvm-as %s 2>&1 | grep "integer constant must have integer type"
; PR2060

define i8* @foo() {
       ret i8* 0
}
