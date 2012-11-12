; RUN: llc < %s -march=ppc32

define i32 @foo() {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%tmp = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp1 = call i32 @llvm.flt.rounds( )		; <i32> [#uses=1]
	store i32 %tmp1, i32* %tmp, align 4
	%tmp2 = load i32* %tmp, align 4		; <i32> [#uses=1]
	store i32 %tmp2, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval3 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval3
}

declare i32 @llvm.flt.rounds() nounwind 
