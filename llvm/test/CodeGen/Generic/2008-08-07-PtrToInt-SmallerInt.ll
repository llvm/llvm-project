; RUN: llc < %s
; PR2603
        %struct.A = type { i8 }
        %struct.B = type { i8, [1 x i8] }
@Foo = constant %struct.A { i8 ptrtoint (i8* getelementptr ([1 x i8]* inttoptr (i32 17 to [1 x i8]*), i32 0, i32 -16) to i8) }          ; <%struct.A*> [#uses=0]
