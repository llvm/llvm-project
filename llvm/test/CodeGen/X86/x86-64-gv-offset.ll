; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | not grep lea

	%struct.x = type { float, double }
@X = global %struct.x { float 1.000000e+00, double 2.000000e+00 }, align 16		; <%struct.x*> [#uses=2]

define i32 @main() nounwind  {
entry:
	%tmp2 = load float* getelementptr (%struct.x* @X, i32 0, i32 0), align 16		; <float> [#uses=1]
	%tmp4 = load double* getelementptr (%struct.x* @X, i32 0, i32 1), align 8		; <double> [#uses=1]
	tail call void @t( float %tmp2, double %tmp4 ) nounwind 
	ret i32 0
}

declare void @t(float, double)
