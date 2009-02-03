; RUN: llvm-as < %s | opt -inline -disable-output
	%struct.Benchmark = type { i32 (...)** }
	%struct.Complex = type { double, double }
	%struct.ComplexBenchmark = type { %struct.Benchmark }

define %struct.Complex @_Zml7ComplexS_1(double %a.0, double %a.1, double %b.0, double %b.1) nounwind  {
entry:
	%mrv = alloca %struct.Complex		; <%struct.Complex*> [#uses=2]
	%mrv.gep = getelementptr %struct.Complex* %mrv, i32 0, i32 0		; <double*> [#uses=1]
	%mrv.ld = load double* %mrv.gep		; <double> [#uses=1]
	%mrv.gep1 = getelementptr %struct.Complex* %mrv, i32 0, i32 1		; <double*> [#uses=1]
	%mrv.ld2 = load double* %mrv.gep1		; <double> [#uses=1]
	ret double %mrv.ld, double %mrv.ld2
}

define void @_ZNK16ComplexBenchmark9oop_styleEv(%struct.ComplexBenchmark* %this) nounwind  {
entry:
	%tmp = alloca %struct.Complex		; <%struct.Complex*> [#uses=0]
	br label %bb31
bb:		; preds = %bb31
	call %struct.Complex @_Zml7ComplexS_1( double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00 ) nounwind 		; <%struct.Complex>:0 [#uses=1]
	%gr = getresult %struct.Complex %0, 1		; <double> [#uses=0]
	br label %bb31
bb31:		; preds = %bb, %entry
	br i1 false, label %bb, label %return
return:		; preds = %bb31
	ret void
}
