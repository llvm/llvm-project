; RUN: llvm-as < %s | opt -basicaa -licm | llvm-dis | %prcontext sin 1 | grep Out:

declare double @sin(double) readnone

declare void @foo()

define double @test(double %X) {
	br label %Loop

Loop:		; preds = %Loop, %0
	call void @foo( )
	%A = call double @sin( double %X ) readnone		; <double> [#uses=1]
	br i1 true, label %Loop, label %Out

Out:		; preds = %Loop
	ret double %A
}
