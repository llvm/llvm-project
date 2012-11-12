; RUN: opt < %s -scalarrepl -S | not grep shr

%struct.S = type { i16 }

define zeroext i1 @f(i16 signext  %b)   {
entry:
	%b_addr = alloca i16		; <i16*> [#uses=2]
	%retval = alloca i32		; <i32*> [#uses=2]
	%s = alloca %struct.S		; <%struct.S*> [#uses=2]
	%tmp = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i16 %b, i16* %b_addr
	%tmp1 = getelementptr %struct.S* %s, i32 0, i32 0		; <i16*> [#uses=1]
	%tmp2 = load i16* %b_addr, align 2		; <i16> [#uses=1]
	store i16 %tmp2, i16* %tmp1, align 2
	%tmp3 = getelementptr %struct.S* %s, i32 0, i32 0		; <i16*> [#uses=1]
	%tmp34 = bitcast i16* %tmp3 to [2 x i1]*		; <[2 x i1]*> [#uses=1]
	%tmp5 = getelementptr [2 x i1]* %tmp34, i32 0, i32 1		; <i1*> [#uses=1]
	%tmp6 = load i1* %tmp5, align 1		; <i1> [#uses=1]
	%tmp67 = zext i1 %tmp6 to i32		; <i32> [#uses=1]
	store i32 %tmp67, i32* %tmp, align 4
	%tmp8 = load i32* %tmp, align 4		; <i32> [#uses=1]
	store i32 %tmp8, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval9 = load i32* %retval		; <i32> [#uses=1]
	%retval910 = trunc i32 %retval9 to i1		; <i1> [#uses=1]
	ret i1 %retval910
}
