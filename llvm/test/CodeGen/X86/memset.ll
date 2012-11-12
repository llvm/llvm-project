; RUN: llc < %s -march=x86 -mattr=-sse -mtriple=i686-apple-darwin8.8.0 | grep mov | count 9
; RUN: llc < %s -march=x86 -mattr=+sse -mtriple=i686-apple-darwin8.8.0 | grep mov | count 3

	%struct.x = type { i16, i16 }

define void @t() nounwind  {
entry:
	%up_mvd = alloca [8 x %struct.x]		; <[8 x %struct.x]*> [#uses=2]
	%up_mvd116 = getelementptr [8 x %struct.x]* %up_mvd, i32 0, i32 0		; <%struct.x*> [#uses=1]
	%tmp110117 = bitcast [8 x %struct.x]* %up_mvd to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.p0i8.i64(i8* %tmp110117, i8 0, i64 32, i32 8, i1 false)
	call void @foo( %struct.x* %up_mvd116 ) nounwind 
	ret void
}

declare void @foo(%struct.x*)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
