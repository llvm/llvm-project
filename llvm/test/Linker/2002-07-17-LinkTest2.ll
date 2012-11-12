; This fails linking when it is linked with an empty file as the first object file

; RUN: llvm-as > %t1.bc < /dev/null
; RUN: llvm-as < %s  > %t2.bc
; RUN: llvm-link %t1.bc %t2.bc

@work = global i32 (i32, i32)* @zip		; <i32 (i32, i32)**> [#uses=0]

declare i32 @zip(i32, i32)

