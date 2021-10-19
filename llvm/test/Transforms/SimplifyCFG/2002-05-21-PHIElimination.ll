; CFG Simplification is making a loop dead, then changing the add into:
;
;   %V1 = add int %V1, 1
;
; Which is not valid SSA
;
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 | llvm-dis

define void @test() {
; <label>:0
	br i1 true, label %end, label %Loop
Loop:		; preds = %Loop, %0
	%V = phi i32 [ 0, %0 ], [ %V1, %Loop ]		; <i32> [#uses=1]
	%V1 = add i32 %V, 1		; <i32> [#uses=1]
	br label %Loop
end:		; preds = %0
	ret void
}

