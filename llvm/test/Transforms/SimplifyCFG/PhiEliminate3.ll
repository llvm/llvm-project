; Test merging of blocks containing complex expressions,
; with various folding thresholds
;
; RUN: opt < %s -simplifycfg -S -phi-node-folding-threshold=1 | grep N:
; RUN: opt < %s -simplifycfg -S -phi-node-folding-threshold=2 | not grep N:
; RUN: opt < %s -simplifycfg -S -phi-node-folding-threshold=2 | grep M:
; RUN: opt < %s -simplifycfg -S -phi-node-folding-threshold=7 | not grep M:
;

define i32 @test(i1 %a, i1 %b, i32 %i, i32 %j, i32 %k) {
entry:
        br i1 %a, label %M, label %O
O:
        br i1 %b, label %P, label %Q
P:
        %iaj = add i32 %i, %j
        %iajak = add i32 %iaj, %k
        br label %N
Q:
        %ixj = xor i32 %i, %j
        %ixjxk = xor i32 %ixj, %k
        br label %N
N:
        ; This phi should be foldable if threshold >= 2
        %Wp = phi i32 [ %iajak, %P ], [ %ixjxk, %Q ]
        %Wp2 = add i32 %Wp, %Wp
        br label %M
M:
        ; This phi should be foldable if threshold >= 7
        %W = phi i32 [ %Wp2, %N ], [ 2, %entry ]
        %R = add i32 %W, 1
        ret i32 %R
}

