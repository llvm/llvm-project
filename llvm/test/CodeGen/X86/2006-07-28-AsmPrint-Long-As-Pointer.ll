; RUN: llc < %s -march=x86 | grep -- 4294967240
; PR853

@X = global i32* inttoptr (i64 -56 to i32*)		; <i32**> [#uses=0]

