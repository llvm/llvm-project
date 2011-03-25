; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s
; CHECK-NOT: movsd
; CHECK: movd {{%rdi|%rcx}}, %xmm0
; CHECK-NOT: movsd

define <2 x i64> @test(i64 %i) nounwind  {
entry:
	%tmp10 = insertelement <2 x i64> undef, i64 %i, i32 0
	%tmp11 = insertelement <2 x i64> %tmp10, i64 0, i32 1
	ret <2 x i64> %tmp11
}

