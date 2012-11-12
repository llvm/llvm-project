; Test merging of blocks that only have PHI nodes in them
;
; RUN: opt < %s -simplifycfg -S | FileCheck %s
;

define i32 @test(i1 %a, i1 %b) {
; CHECK: br i1 %a
        br i1 %a, label %M, label %O
; CHECK: O:
O:              ; preds = %0
; CHECK: select i1 %b, i32 0, i32 1
; CHECK-NOT: phi
        br i1 %b, label %N, label %Q
Q:              ; preds = %O
        br label %N
N:              ; preds = %Q, %O
        ; This block should be foldable into M
        %Wp = phi i32 [ 0, %O ], [ 1, %Q ]              ; <i32> [#uses=1]
        br label %M
M:              ; preds = %N, %0
; CHECK: %W = phi i32
        %W = phi i32 [ %Wp, %N ], [ 2, %0 ]             ; <i32> [#uses=1]
        %R = add i32 %W, 1              ; <i32> [#uses=1]
        ret i32 %R
}

