; RUN: llc < %s -march=arm | grep movne | count 1
; RUN: llc < %s -march=arm | grep moveq | count 1

define i32 @f1(float %X, float %Y) {
	%tmp = fcmp uno float %X, %Y
	%retval = select i1 %tmp, i32 1, i32 -1
	ret i32 %retval
}

define i32 @f2(float %X, float %Y) {
	%tmp = fcmp ord float %X, %Y
	%retval = select i1 %tmp, i32 1, i32 -1
	ret i32 %retval
}
