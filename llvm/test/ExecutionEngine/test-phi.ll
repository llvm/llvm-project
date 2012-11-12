; RUN: %lli %s > /dev/null

; test phi node
@Y = global i32 6		; <i32*> [#uses=1]

define void @blah(i32* %X) {
; <label>:0
	br label %T
T:		; preds = %Dead, %0
	phi i32* [ %X, %0 ], [ @Y, %Dead ]		; <i32*>:1 [#uses=0]
	ret void
Dead:		; No predecessors!
	br label %T
}

define i32 @test(i1 %C) {
; <label>:0
	br i1 %C, label %T, label %T
T:		; preds = %0, %0
	%X = phi i32 [ 123, %0 ], [ 123, %0 ]		; <i32> [#uses=1]
	ret i32 %X
}

define i32 @main() {
; <label>:0
	br label %Test
Test:		; preds = %Dead, %0
	%X = phi i32 [ 0, %0 ], [ %Y, %Dead ]		; <i32> [#uses=1]
	ret i32 %X
Dead:		; No predecessors!
	%Y = ashr i32 12, 4		; <i32> [#uses=1]
	br label %Test
}

