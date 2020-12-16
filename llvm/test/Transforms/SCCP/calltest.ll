; RUN: opt < %s -sccp -loop-deletion -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

declare double @sqrt(double) readnone nounwind
%empty = type {}
declare %empty @has_side_effects()

define double @test_0(i32 %param) {
; CHECK-LABEL: @test_0(
; CHECK-NOT: br
entry:
; No matter how hard you try, sqrt(1.0) is always 1.0.  This allows the
; optimizer to delete this loop.

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

define i32 @test_1() {
; CHECK-LABEL: @test_1(
; CHECK: call %empty @has_side_effects()
  %1 = call %empty @has_side_effects()
  ret i32 0
}
