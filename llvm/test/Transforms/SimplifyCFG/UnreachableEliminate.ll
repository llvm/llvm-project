; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep unreachable

define void @test1(i1 %C, i1* %BP) {
        br i1 %C, label %T, label %F
T:              ; preds = %0
        store i1 %C, i1* %BP
        unreachable
F:              ; preds = %0
        ret void
}

define void @test2() {
        invoke void @test2( )
                        to label %N unwind label %U
U:              ; preds = %0
        unreachable
N:              ; preds = %0
        ret void
}

define i32 @test3(i32 %v) {
        switch i32 %v, label %default [
                 i32 1, label %U
                 i32 2, label %T
        ]
default:                ; preds = %0
        ret i32 1
U:              ; preds = %0
        unreachable
T:              ; preds = %0
        ret i32 2
}

