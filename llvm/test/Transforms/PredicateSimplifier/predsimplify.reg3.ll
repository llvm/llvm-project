; RUN: llvm-as < %s | opt -predsimplify -simplifycfg | llvm-dis | grep pass

define void @regtest(i32 %x) {
entry:
	%A = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %A, label %middle, label %after
middle:		; preds = %entry
	br label %after
after:		; preds = %middle, %entry
	%B = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br i1 %B, label %then, label %else
then:		; preds = %after
	br label %end
else:		; preds = %after
	call void (...)* @pass( )
	br label %end
end:		; preds = %else, %then
	ret void
}

declare void @pass(...)

