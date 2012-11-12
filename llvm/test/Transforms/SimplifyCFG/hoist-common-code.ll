; RUN: opt < %s -simplifycfg -S | not grep br

declare void @bar(i32)

define void @test(i1 %P, i32* %Q) {
        br i1 %P, label %T, label %F
T:              ; preds = %0
        store i32 1, i32* %Q
        %A = load i32* %Q               ; <i32> [#uses=1]
        call void @bar( i32 %A )
        ret void
F:              ; preds = %0
        store i32 1, i32* %Q
        %B = load i32* %Q               ; <i32> [#uses=1]
        call void @bar( i32 %B )
        ret void
}

