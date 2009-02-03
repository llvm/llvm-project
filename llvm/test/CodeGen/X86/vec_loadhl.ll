; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movlpd
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movhpd
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep movsd

define void @t1(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
	%tmp3 = load <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 2, i32 1 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
}

define void @t2(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
	%tmp3 = load <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 0, i32 2 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
}
