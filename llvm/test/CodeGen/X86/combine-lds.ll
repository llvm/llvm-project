; RUN: llc < %s -march=x86 -mattr=+sse2 | grep fldl | count 1

define double @doload64(i64 %x) nounwind  {
	%tmp717 = bitcast i64 %x to double
	ret double %tmp717
}
