; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep xori

define i32 @test(i1 %B, ptr %P) {
        br i1 %B, label %T, label %F

T:              ; preds = %0
        store i32 123, ptr %P
        ret i32 0

F:              ; preds = %0
        ret i32 17
}

