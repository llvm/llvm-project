; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movq
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep mov | count 1
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+sse2 | grep movd

define <2 x i64> @t1(i64 %x) nounwind  {
	%tmp8 = insertelement <2 x i64> zeroinitializer, i64 %x, i32 0
	ret <2 x i64> %tmp8
}
