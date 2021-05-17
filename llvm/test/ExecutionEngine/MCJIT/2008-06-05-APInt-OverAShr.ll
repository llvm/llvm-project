; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s | FileCheck %s
; CHECK: 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
@.str = internal constant [10 x i8] c"MSB = %d\0A\00"		; <[10 x i8]*> [#uses=1]

define i65 @foo(i65 %x) {
entry:
	%x_addr = alloca i65		; <i65*> [#uses=2]
	%retval = alloca i65		; <i65*> [#uses=2]
	%tmp = alloca i65		; <i65*> [#uses=2]
	%"alloca point" = bitcast i65 0 to i65		; <i65> [#uses=0]
	store i65 %x, i65* %x_addr
	%tmp1 = load i65, i65* %x_addr, align 4		; <i65> [#uses=1]
	%tmp2 = ashr i65 %tmp1, 65		; <i65> [#uses=1]
	store i65 %tmp2, i65* %tmp, align 4
	%tmp3 = load i65, i65* %tmp, align 4		; <i65> [#uses=1]
	store i65 %tmp3, i65* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval4 = load i65, i65* %retval		; <i65> [#uses=1]
	ret i65 %retval4
}

define i32 @main() {
entry:
	%retval = alloca i32		; <i32*> [#uses=1]
	%iftmp.0 = alloca i32		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = call i65 @foo( i65 -9 )		; <i65> [#uses=1]
	%tmp1 = lshr i65 %tmp, 64		; <i65> [#uses=1]
	%tmp2 = xor i65 %tmp1, 1		; <i65> [#uses=1]
	%tmp3 = and i65 %tmp2, 1		; <i65> [#uses=1]
	%tmp34 = trunc i65 %tmp3 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp34, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	store i32 0, i32* %iftmp.0, align 4
	br label %cond_next

cond_false:		; preds = %entry
	store i32 1, i32* %iftmp.0, align 4
	br label %cond_next

cond_next:		; preds = %cond_false, %cond_true
	%tmp5 = getelementptr [10 x i8], [10 x i8]* @.str, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp6 = load i32, i32* %iftmp.0, align 4		; <i32> [#uses=1]
	%tmp7 = call i32 (i8*, ...) @printf( i8* noalias  %tmp5, i32 %tmp6 ) nounwind 		; <i32> [#uses=0]
	br label %return

return:		; preds = %cond_next
    store i32 0, i32* %retval, align 4
	%retval8 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval8
}

declare i32 @printf(i8* noalias , ...) nounwind 
