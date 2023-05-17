; PR1117
; RUN: not llvm-as %s -o /dev/null 2>&1 | grep "invalid cast opcode for cast from"

define ptr @nada(i64 %X) {
    %result = trunc i64 %X to ptr
    ret ptr %result
}
