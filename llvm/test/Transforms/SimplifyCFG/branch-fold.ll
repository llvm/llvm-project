; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep {br i1} | count 1

define void @test(i32* %P, i32* %Q, i1 %A, i1 %B) {
        br i1 %A, label %a, label %b
a:              ; preds = %0
        br i1 %B, label %b, label %c
b:              ; preds = %a, %0
        store i32 123, i32* %P
        ret void
c:              ; preds = %a
        ret void
}

