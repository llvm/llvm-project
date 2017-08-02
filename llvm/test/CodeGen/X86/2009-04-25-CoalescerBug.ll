; RUN: llc < %s -mtriple=x86_64-- | grep mov | count 1
; rdar://6806252

define i64 @test(i32* %tmp13) nounwind {
entry:
	br label %while.cond

while.cond:		; preds = %while.cond, %entry
	%tmp15 = load i32, i32* %tmp13		; <i32> [#uses=2]
	%bf.lo = lshr i32 %tmp15, 1		; <i32> [#uses=1]
	%bf.lo.cleared = and i32 %bf.lo, 2147483647		; <i32> [#uses=1]
	%conv = zext i32 %bf.lo.cleared to i64		; <i64> [#uses=1]
	%bf.lo.cleared25 = and i32 %tmp15, 1		; <i32> [#uses=1]
	%tobool = icmp ne i32 %bf.lo.cleared25, 0		; <i1> [#uses=1]
	br i1 %tobool, label %while.cond, label %while.end

while.end:		; preds = %while.cond
	ret i64 %conv
}
