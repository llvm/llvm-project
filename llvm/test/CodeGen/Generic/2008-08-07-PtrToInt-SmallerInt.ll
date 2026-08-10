; RUN: llc < %s
; PR2603
        %struct.A = type { i8 }
        %struct.B = type { i8, [1 x i8] }
@Foo = constant %struct.A { i8 ptrtoint (ptr getelementptr ([1 x i8], ptr inttoptr (i32 17 to ptr), i32 0, i32 -16) to i8) }          ; <ptr> [#uses=0]
