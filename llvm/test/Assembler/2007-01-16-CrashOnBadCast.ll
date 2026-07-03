; PR1117
; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: invalid cast opcode for cast from 'i64' to 'ptr'

define ptr @nada(i64 %X) {
    %result = trunc i64 %X to ptr
    ret ptr %result
}
