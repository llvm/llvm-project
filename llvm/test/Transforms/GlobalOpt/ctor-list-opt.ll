; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep CTOR
@llvm.global_ctors = appending global [10 x { i32, void ()* }] [ { i32, void ()* } { i32 65535, void ()* @CTOR1 }, { i32, void ()* } { i32 65535, void ()* @CTOR1 }, { i32, void ()* } { i32 65535, void ()* @CTOR2 }, { i32, void ()* } { i32 65535, void ()* @CTOR3 }, { i32, void ()* } { i32 65535, void ()* @CTOR4 }, { i32, void ()* } { i32 65535, void ()* @CTOR5 }, { i32, void ()* } { i32 65535, void ()* @CTOR6 }, { i32, void ()* } { i32 65535, void ()* @CTOR7 }, { i32, void ()* } { i32 65535, void ()* @CTOR8 }, { i32, void ()* } { i32 2147483647, void ()* null } ]		; <[10 x { i32, void ()* }]*> [#uses=0]
@G = global i32 0		; <i32*> [#uses=1]
@G2 = global i32 0		; <i32*> [#uses=1]
@G3 = global i32 -123		; <i32*> [#uses=2]
@X = global { i32, [2 x i32] } { i32 0, [2 x i32] [ i32 17, i32 21 ] }		; <{ i32, [2 x i32] }*> [#uses=2]
@Y = global i32 -1		; <i32*> [#uses=2]
@Z = global i32 123		; <i32*> [#uses=1]
@D = global double 0.000000e+00		; <double*> [#uses=1]
@CTORGV = internal global i1 false		; <i1*> [#uses=2]

define internal void @CTOR1() {
	ret void
}

define internal void @CTOR2() {
	%A = add i32 1, 23		; <i32> [#uses=1]
	store i32 %A, i32* @G
	store i1 true, i1* @CTORGV
	ret void
}

define internal void @CTOR3() {
	%X = or i1 true, false		; <i1> [#uses=1]
	br label %Cont

Cont:		; preds = %0
	br i1 %X, label %S, label %T

S:		; preds = %Cont
	store i32 24, i32* @G2
	ret void

T:		; preds = %Cont
	ret void
}

define internal void @CTOR4() {
	%X = load i32* @G3		; <i32> [#uses=1]
	%Y = add i32 %X, 123		; <i32> [#uses=1]
	store i32 %Y, i32* @G3
	ret void
}

define internal void @CTOR5() {
	%X.2p = getelementptr { i32, [2 x i32] }* @X, i32 0, i32 1, i32 0		; <i32*> [#uses=2]
	%X.2 = load i32* %X.2p		; <i32> [#uses=1]
	%X.1p = getelementptr { i32, [2 x i32] }* @X, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %X.2, i32* %X.1p
	store i32 42, i32* %X.2p
	ret void
}

define internal void @CTOR6() {
	%A = alloca i32		; <i32*> [#uses=2]
	%y = load i32* @Y		; <i32> [#uses=1]
	store i32 %y, i32* %A
	%Av = load i32* %A		; <i32> [#uses=1]
	%Av1 = add i32 %Av, 1		; <i32> [#uses=1]
	store i32 %Av1, i32* @Y
	ret void
}

define internal void @CTOR7() {
	call void @setto( i32* @Z, i32 0 )
	ret void
}

define void @setto(i32* %P, i32 %V) {
	store i32 %V, i32* %P
	ret void
}

declare double @cos(double)

define internal void @CTOR8() {
	%X = call double @cos( double 1.000000e+00 )		; <double> [#uses=1]
	store double %X, double* @D
	ret void
}

define i1 @accessor() {
	%V = load i1* @CTORGV		; <i1> [#uses=1]
	ret i1 %V
}
