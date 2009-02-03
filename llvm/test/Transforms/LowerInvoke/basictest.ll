; RUN: llvm-as < %s | opt -lowerinvoke -disable-output -enable-correct-eh-support


define i32 @foo() {
	invoke i32 @foo( )
			to label %Ok unwind label %Crap		; <i32>:1 [#uses=0]
Ok:		; preds = %0
	invoke i32 @foo( )
			to label %Ok2 unwind label %Crap		; <i32>:2 [#uses=0]
Ok2:		; preds = %Ok
	ret i32 2
Crap:		; preds = %Ok, %0
	ret i32 1
}

define i32 @bar(i32 %blah) {
	br label %doit
doit:		; preds = %0
        ;; Value live across an unwind edge.
	%B2 = add i32 %blah, 1		; <i32> [#uses=1]
	invoke i32 @foo( )
			to label %Ok unwind label %Crap		; <i32>:1 [#uses=0]
Ok:		; preds = %doit
	invoke i32 @foo( )
			to label %Ok2 unwind label %Crap		; <i32>:2 [#uses=0]
Ok2:		; preds = %Ok
	ret i32 2
Crap:		; preds = %Ok, %doit
	ret i32 %B2
}
