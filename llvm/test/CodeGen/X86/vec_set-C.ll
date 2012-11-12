; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu -mattr=+sse2 | grep movq
; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu -mattr=+sse2 | grep mov | count 1
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-pc-linux -mattr=+sse2 | grep movd

define <2 x i64> @t1(i64 %x) nounwind  {
	%tmp8 = insertelement <2 x i64> zeroinitializer, i64 %x, i32 0
	ret <2 x i64> %tmp8
}
