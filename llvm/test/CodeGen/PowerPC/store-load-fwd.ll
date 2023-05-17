; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep lwz

define i32 @test(ptr %P) {
        store i32 1, ptr %P
        %V = load i32, ptr %P               ; <i32> [#uses=1]
        ret i32 %V
}

