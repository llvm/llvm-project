; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movd
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep xorp

define void @t1() nounwind  {
	%tmp298.i.i = load <4 x float>* null, align 16
	%tmp304.i.i = bitcast <4 x float> %tmp298.i.i to <4 x i32>
	%tmp305.i.i = and <4 x i32> %tmp304.i.i, < i32 -1, i32 0, i32 0, i32 0 >
	store <4 x i32> %tmp305.i.i, <4 x i32>* null, align 16
	unreachable
}
