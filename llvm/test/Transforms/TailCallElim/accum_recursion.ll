; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

define i32 @factorial(i32 %x) {
entry:
	%tmp.1 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then, label %else
then:		; preds = %entry
	%tmp.6 = add i32 %x, -1		; <i32> [#uses=1]
	%tmp.4 = call i32 @factorial( i32 %tmp.6 )		; <i32> [#uses=1]
	%tmp.7 = mul i32 %tmp.4, %x		; <i32> [#uses=1]
	ret i32 %tmp.7
else:		; preds = %entry
	ret i32 1
}

