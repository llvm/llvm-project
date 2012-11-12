; RUN: llc < %s -march=x86 -mattr=+sse2 | grep movsd | count 1

define <2 x i64> @test(<2 x i64>* %p) nounwind {
	%tmp = bitcast <2 x i64>* %p to double*		
	%tmp.upgrd.1 = load double* %tmp	
	%tmp.upgrd.2 = insertelement <2 x double> undef, double %tmp.upgrd.1, i32 0
	%tmp5 = insertelement <2 x double> %tmp.upgrd.2, double 0.0, i32 1
	%tmp.upgrd.3 = bitcast <2 x double> %tmp5 to <2 x i64>
	ret <2 x i64> %tmp.upgrd.3
}

