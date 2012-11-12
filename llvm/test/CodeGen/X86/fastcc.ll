; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+sse2 -post-RA-scheduler=false | FileCheck %s
; CHECK: movsd %xmm0, 8(%esp)
; CHECK: xorl %ecx, %ecx

@d = external global double		; <double*> [#uses=1]
@c = external global double		; <double*> [#uses=1]
@b = external global double		; <double*> [#uses=1]
@a = external global double		; <double*> [#uses=1]

define i32 @foo() nounwind {
entry:
	%0 = load double* @d, align 8		; <double> [#uses=1]
	%1 = load double* @c, align 8		; <double> [#uses=1]
	%2 = load double* @b, align 8		; <double> [#uses=1]
	%3 = load double* @a, align 8		; <double> [#uses=1]
	tail call fastcc void @bar( i32 0, i32 1, i32 2, double 1.000000e+00, double %3, double %2, double %1, double %0 ) nounwind
	ret i32 0
}

declare fastcc void @bar(i32, i32, i32, double, double, double, double, double)
