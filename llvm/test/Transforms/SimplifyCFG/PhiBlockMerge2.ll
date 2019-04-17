; Test merging of blocks that only have PHI nodes in them.  This tests the case
; where the mergedinto block doesn't have any PHI nodes, and is in fact 
; dominated by the block-to-be-eliminated
;
; RUN: opt < %s -simplifycfg -S | not grep N:
;

declare i1 @foo()

define i32 @test(i1 %a, i1 %b) {
        %c = call i1 @foo()
	br i1 %c, label %N, label %P
P:
        %d = call i1 @foo()
	br i1 %d, label %N, label %Q
Q:
	br label %N
N:
	%W = phi i32 [0, %0], [1, %Q], [2, %P]
	; This block should be foldable into M
	br label %M

M:
	%R = add i32 %W, 1
	ret i32 %R
}

