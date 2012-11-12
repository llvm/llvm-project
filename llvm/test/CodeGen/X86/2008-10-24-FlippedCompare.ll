; RUN: llc < %s -march=x86 -mattr=+sse2 -o - | not grep "ucomiss[^,]*esp"

define void @f(float %wt) {
entry:
	%0 = fcmp ogt float %wt, 0.000000e+00		; <i1> [#uses=1]
	%1 = tail call i32 @g(i32 44)		; <i32> [#uses=3]
	%2 = inttoptr i32 %1 to i8*		; <i8*> [#uses=2]
	br i1 %0, label %bb, label %bb1

bb:		; preds = %entry
	ret void

bb1:		; preds = %entry
	ret void
}

declare i32 @g(i32)
