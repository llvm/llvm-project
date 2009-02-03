; Test merging of blocks that only have PHI nodes in them
;
; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep N:
;

define i32 @test(i1 %a, i1 %b) {
; <label>:0
        br i1 %a, label %M, label %O
O:              ; preds = %0
        br i1 %b, label %N, label %Q
Q:              ; preds = %O
        br label %N
N:              ; preds = %Q, %O
        ; This block should be foldable into M
        %Wp = phi i32 [ 0, %O ], [ 1, %Q ]              ; <i32> [#uses=1]
        br label %M
M:              ; preds = %N, %0
        %W = phi i32 [ %Wp, %N ], [ 2, %0 ]             ; <i32> [#uses=1]
        %R = add i32 %W, 1              ; <i32> [#uses=1]
        ret i32 %R
}

