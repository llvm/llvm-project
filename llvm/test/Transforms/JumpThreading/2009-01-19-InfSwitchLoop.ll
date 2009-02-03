; RUN: llvm-as < %s | opt -jump-threading | llvm-dis
; PR3353

define i32 @test(i8 %X) {
entry:
	%Y = add i8 %X, 1
	%Z = add i8 %Y, 1
	br label %bb33.i

bb33.i:		; preds = %bb33.i, %bb32.i
	switch i8 %Y, label %bb32.i [
		i8 39, label %bb35.split.i
		i8 13, label %bb33.i
	]

bb35.split.i:
	ret i32 5
bb32.i:
	ret i32 1
}

