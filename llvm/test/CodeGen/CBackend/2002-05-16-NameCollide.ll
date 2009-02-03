; RUN: llvm-as < %s | llc -march=c

; Make sure that global variables do not collide if they have the same name,
; but different types.

@X = global i32 5               ; <i32*> [#uses=0]
@X.upgrd.1 = global i64 7               ; <i64*> [#uses=0]

