; RUN: llc < %s -mtriple=i686--

define i32 @test() {
	br i1 false, label %cond_next33, label %cond_true12
cond_true12:		; preds = %0
	ret i32 0
cond_next33:		; preds = %0
	%tmp44.i = call double @foo( double 0.000000e+00, i32 32 )		; <double> [#uses=1]
	%tmp61.i = load i8, ptr null		; <i8> [#uses=1]
	%tmp61.i.upgrd.1 = zext i8 %tmp61.i to i32		; <i32> [#uses=1]
	%tmp58.i = or i32 0, %tmp61.i.upgrd.1		; <i32> [#uses=1]
	%tmp62.i = or i32 %tmp58.i, 0		; <i32> [#uses=1]
	%tmp62.i.upgrd.2 = sitofp i32 %tmp62.i to double		; <double> [#uses=1]
	%tmp64.i = fadd double %tmp62.i.upgrd.2, %tmp44.i		; <double> [#uses=1]
	%tmp68.i = call double @foo( double %tmp64.i, i32 0 )		; <double> [#uses=0]
	ret i32 0
}

declare double @foo(double, i32)

