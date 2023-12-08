; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep addi

        %struct.X = type { [5 x i8] }

define i32 @test1(ptr %P, i32 %i) {
        %tmp.2 = add i32 %i, 2          ; <i32> [#uses=1]
        %tmp.4 = getelementptr [4 x i32], ptr %P, i32 %tmp.2, i32 1         ; <ptr> [#uses=1]
        %tmp.5 = load i32, ptr %tmp.4               ; <i32> [#uses=1]
        ret i32 %tmp.5
}

define i32 @test2(ptr %P, i32 %i) {
        %tmp.2 = add i32 %i, 2          ; <i32> [#uses=1]
        %tmp.5 = getelementptr %struct.X, ptr %P, i32 %tmp.2, i32 0, i32 1          ; <ptr> [#uses=1]
        %tmp.6 = load i8, ptr %tmp.5                ; <i8> [#uses=1]
        %tmp.7 = sext i8 %tmp.6 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.7
}

