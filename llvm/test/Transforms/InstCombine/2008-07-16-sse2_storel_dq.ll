; RUN: opt < %s -instcombine -S | not grep "store "
; PR2296

@G = common global double 0.000000e+00, align 16

define void @x(<2 x i64> %y) nounwind  {
entry:
	bitcast <2 x i64> %y to <4 x i32>
	call void @llvm.x86.sse2.storel.dq( i8* bitcast (double* @G to i8*), <4 x i32> %0 ) nounwind 
	ret void
}

declare void @llvm.x86.sse2.storel.dq(i8*, <4 x i32>) nounwind 
