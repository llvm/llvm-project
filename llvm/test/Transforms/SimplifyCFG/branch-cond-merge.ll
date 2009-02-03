; RUN: llvm-as < %s | opt -simplifycfg -instcombine \
; RUN:   -simplifycfg | llvm-dis | not grep call

declare void @bar()

define void @test(i32 %X, i32 %Y) {
entry:
        %tmp.2 = icmp ne i32 %X, %Y             ; <i1> [#uses=1]
        br i1 %tmp.2, label %shortcirc_next, label %UnifiedReturnBlock
shortcirc_next:         ; preds = %entry
        %tmp.3 = icmp ne i32 %X, %Y             ; <i1> [#uses=1]
        br i1 %tmp.3, label %UnifiedReturnBlock, label %then
then:           ; preds = %shortcirc_next
        call void @bar( )
        ret void
UnifiedReturnBlock:             ; preds = %shortcirc_next, %entry
        ret void
}

