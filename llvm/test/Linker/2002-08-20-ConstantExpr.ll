; This fails linking when it is linked with an empty file as the first object file

; RUN: llvm-as > %t.LinkTest.bc < /dev/null
; RUN: llvm-as < %s > %t.bc
; RUN: llvm-link %t.LinkTest.bc %t.bc

@work = global i32 4		; <i32*> [#uses=1]
@test = global i32* getelementptr (i32* @work, i64 1)		; <i32**> [#uses=0]

