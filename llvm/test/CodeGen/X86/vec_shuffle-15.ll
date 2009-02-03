; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

define <2 x i64> @t00(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 0, i32 0 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t01(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 0, i32 1 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t02(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 0, i32 2 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t03(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 0, i32 3 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t10(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 1, i32 0 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t11(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 1, i32 1 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t12(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 1, i32 2 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t13(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 1, i32 3 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t20(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 2, i32 0 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t21(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 2, i32 1 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t22(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 2, i32 2 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t23(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 2, i32 3 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t30(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 3, i32 0 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t31(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 3, i32 1 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t32(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 3, i32 2 >
	ret <2 x i64> %tmp
}

define <2 x i64> @t33(<2 x i64> %a, <2 x i64> %b) nounwind  {
	%tmp = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> < i32 3, i32 3 >
	ret <2 x i64> %tmp
}
