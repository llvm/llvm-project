; PR1117
; RUN: not llvm-as %s -o /dev/null 2>&1 | grep "invalid cast opcode for cast from"

@X = constant ptr trunc (i64 0 to ptr)
