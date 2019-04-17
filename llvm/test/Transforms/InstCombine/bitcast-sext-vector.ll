; RUN: opt < %s -instcombine -S | FileCheck %s
; CHECK: sext
; Don't fold zero/sign extensions with a bitcast between a vector and scalar.

define i32 @t(<4 x i8> %src1, <4 x i8> %src2) nounwind readonly {
entry:
	%cmp = icmp eq <4 x i8> %src1, %src2; <<4 x i1>> [#uses=1]
	%sext = sext <4 x i1> %cmp to <4 x i8>
	%val = bitcast <4 x i8> %sext to i32
	ret i32 %val
}
