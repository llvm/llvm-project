; The loop sinker was running from the bottom of the loop to the top, causing
; it to miss opportunities to sink instructions that depended on sinking other
; instructions from the loop.  Instead they got hoisted, which is better than
; leaving them in the loop, but increases register pressure pointlessly.

; RUN: llvm-as < %s | opt -licm | llvm-dis | \
; RUN:    %prcontext getelementptr 1 | grep Out:

	%Ty = type { i32, i32 }
@X = external global %Ty		; <%Ty*> [#uses=1]

define i32 @test() {
	br label %Loop
Loop:		; preds = %Loop, %0
	%dead = getelementptr %Ty* @X, i64 0, i32 0		; <i32*> [#uses=1]
	%sunk2 = load i32* %dead		; <i32> [#uses=1]
	br i1 false, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %sunk2
}

