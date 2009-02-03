; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

define void @foo(i1 %C, i32* %P) {
        br i1 %C, label %T, label %F
T:              ; preds = %0
        store i32 7, i32* %P
        ret void
F:              ; preds = %0
        store i32 7, i32* %P
        ret void
}
