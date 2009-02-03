; RUN: llvm-as < %s | opt -inline -f - | llvm-dis | grep "define internal i32 @bar" 
@llvm.noinline = appending global [1 x i8*] [ i8* bitcast (i32 (i32, i32)* @bar to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define internal i32 @bar(i32 %x, i32 %y) {
entry:
	%x_addr = alloca i32		; <i32*> [#uses=2]
	%y_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %x, i32* %x_addr
	store i32 %y, i32* %y_addr
	%tmp1 = load i32* %x_addr		; <i32> [#uses=1]
	%tmp2 = load i32* %y_addr		; <i32> [#uses=1]
	%tmp3 = add i32 %tmp1, %tmp2		; <i32> [#uses=1]
	store i32 %tmp3, i32* %tmp
	%tmp4 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp4, i32* %retval
	br label %return

return:		; preds = %entry
	%retval5 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval5
}

define i32 @foo(i32 %a, i32 %b) {
entry:
	%a_addr = alloca i32		; <i32*> [#uses=2]
	%b_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %a, i32* %a_addr
	store i32 %b, i32* %b_addr
	%tmp1 = load i32* %b_addr		; <i32> [#uses=1]
	%tmp2 = load i32* %a_addr		; <i32> [#uses=1]
	%tmp3 = call i32 @bar( i32 %tmp1, i32 %tmp2 )		; <i32> [#uses=1]
	store i32 %tmp3, i32* %tmp
	%tmp4 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp4, i32* %retval
	br label %return

return:		; preds = %entry
	%retval5 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval5
}
