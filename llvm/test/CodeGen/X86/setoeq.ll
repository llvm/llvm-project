; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 | FileCheck %s

define zeroext i8 @t(double %x) nounwind readnone {
entry:
	%0 = fptosi double %x to i32		; <i32> [#uses=1]
	%1 = sitofp i32 %0 to double		; <double> [#uses=1]
	%2 = fcmp oeq double %1, %x		; <i1> [#uses=1]
	%retval12 = zext i1 %2 to i8		; <i8> [#uses=1]
; CHECK: cmpeqsd
	ret i8 %retval12
}

define zeroext i8 @u(double %x) nounwind readnone {
entry:
	%0 = fptosi double %x to i32		; <i32> [#uses=1]
	%1 = sitofp i32 %0 to double		; <double> [#uses=1]
	%2 = fcmp une double %1, %x		; <i1> [#uses=1]
	%retval12 = zext i1 %2 to i8		; <i8> [#uses=1]
; CHECK: cmpneqsd
	ret i8 %retval12
}
