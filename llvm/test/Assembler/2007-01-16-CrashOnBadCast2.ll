; PR1117
; RUN: not llvm-as %s -o /dev/null 2>&1 | grep "invalid cast opcode for cast from"

@X = constant i8* trunc (i64 0 to i8*)
