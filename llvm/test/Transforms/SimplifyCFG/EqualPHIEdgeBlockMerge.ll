; Test merging of blocks with phi nodes.
;
; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep N:
;

define i32 @test(i1 %a) {
Q:
        br i1 %a, label %N, label %M
N:              ; preds = %Q
        br label %M
M:              ; preds = %N, %Q
        ; It's ok to merge N and M because the incoming values for W are the
        ; same for both cases...
        %W = phi i32 [ 2, %N ], [ 2, %Q ]               ; <i32> [#uses=1]
        %R = add i32 %W, 1              ; <i32> [#uses=1]
        ret i32 %R
}

