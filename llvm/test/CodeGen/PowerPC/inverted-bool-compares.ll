; RUN: llvm-as < %s | llc -march=ppc32 | not grep xori

define i32 @test(i1 %B, i32* %P) {
        br i1 %B, label %T, label %F

T:              ; preds = %0
        store i32 123, i32* %P
        ret i32 0

F:              ; preds = %0
        ret i32 17
}

