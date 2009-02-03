; RUN: llvm-as < %s | opt -sccp -loop-deletion -simplifycfg | llvm-dis | \
; RUN:   not grep br

; No matter how hard you try, sqrt(1.0) is always 1.0.  This allows the
; optimizer to delete this loop.

declare double @sqrt(double)

define double @test(i32 %param) {
entry:
	br label %Loop
Loop:		; preds = %Loop, %entry
	%I2 = phi i32 [ 0, %entry ], [ %I3, %Loop ]		; <i32> [#uses=1]
	%V = phi double [ 1.000000e+00, %entry ], [ %V2, %Loop ]		; <double> [#uses=2]
	%V2 = call double @sqrt( double %V )		; <double> [#uses=1]
	%I3 = add i32 %I2, 1		; <i32> [#uses=2]
	%tmp.7 = icmp ne i32 %I3, %param		; <i1> [#uses=1]
	br i1 %tmp.7, label %Loop, label %Exit
Exit:		; preds = %Loop
	ret double %V
}

