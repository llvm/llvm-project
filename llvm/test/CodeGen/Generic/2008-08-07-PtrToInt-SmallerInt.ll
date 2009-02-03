; RUN: llvm-as < %s | llc
; PR2603
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
        %struct.A = type { i8 }
        %struct.B = type { i8, [1 x i8] }
@Foo = constant %struct.A { i8 ptrtoint (i8* getelementptr ([1 x i8]* inttoptr (i32 17 to [1 x i8]*), i32 0, i32 -16) to i8) }          ; <%struct.A*> [#uses=0]
