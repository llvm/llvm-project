; RUN: llvm-as < %s | llc -march=c
@y = weak global i8 0           ; <i8*> [#uses=1]

define i32 @testcaseshr() {
entry:
        ret i32 lshr (i32 ptrtoint (i8* @y to i32), i32 4)
}

define i32 @testcaseshl() {
entry:
        ret i32 shl (i32 ptrtoint (i8* @y to i32), i32 4)
}

