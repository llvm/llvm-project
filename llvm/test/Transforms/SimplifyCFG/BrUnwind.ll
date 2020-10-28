; RUN: opt < %s -simplifycfg -hoist-common-insts=true -S | \
; RUN: not grep "br label"

define void @test(i1 %C) {
        br i1 %C, label %A, label %B
A:              ; preds = %0
        call void @test( i1 %C )
        br label %X
B:              ; preds = %0
        call void @test( i1 %C )
        br label %X
X:              ; preds = %B, %A
        ret void
}

