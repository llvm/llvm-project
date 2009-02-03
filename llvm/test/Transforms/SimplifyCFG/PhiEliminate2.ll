; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

define i32 @test(i1 %C, i32 %V1, i32 %V2) {
entry:
        br i1 %C, label %then, label %Cont
then:           ; preds = %entry
        %V3 = or i32 %V2, %V1           ; <i32> [#uses=1]
        br label %Cont
Cont:           ; preds = %then, %entry
        %V4 = phi i32 [ %V1, %entry ], [ %V3, %then ]           ; <i32> [#uses=0]
        call i32 @test( i1 false, i32 0, i32 0 )                ; <i32>:0 [#uses=0]
        ret i32 %V1
}

