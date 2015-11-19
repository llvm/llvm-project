; PR1226
; RUN: opt < %s -scalarrepl -S | grep "ret i32 16843009"
; RUN: opt < %s -scalarrepl -S | not grep alloca
; RUN: opt < %s -scalarrepl -instcombine -S | grep "ret i16 514"

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
	%struct.bar = type { %struct.foo, i64, double }
	%struct.foo = type { i32, i32 }


define i32 @test1(%struct.foo* %P) {
entry:
	%L = alloca %struct.foo, align 8		; <%struct.foo*> [#uses=2]
	%L2 = bitcast %struct.foo* %L to i8*		; <i8*> [#uses=1]
	%tmp13 = bitcast %struct.foo* %P to i8*		; <i8*> [#uses=1]
        call void @llvm.memcpy.p0i8.p0i8.i32(i8* %L2, i8* %tmp13, i32 8, i32 4, i1 false)
	%tmp4 = getelementptr %struct.foo, %struct.foo* %L, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp5 = load i32, i32* %tmp4		; <i32> [#uses=1]
	ret i32 %tmp5
}


define i32 @test2() {
entry:
	%L = alloca [4 x %struct.foo], align 16		; <[4 x %struct.foo]*> [#uses=2]
	%L12 = bitcast [4 x %struct.foo]* %L to i8*		; <i8*> [#uses=1]
        call void @llvm.memset.p0i8.i32(i8* %L12, i8 0, i32 32, i32 16, i1 false)
	%tmp4 = getelementptr [4 x %struct.foo], [4 x %struct.foo]* %L, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp5 = load i32, i32* %tmp4		; <i32> [#uses=1]
	ret i32 %tmp5
}


define i32 @test3() {
entry:
	%B = alloca %struct.bar, align 16		; <%struct.bar*> [#uses=4]
	%B1 = bitcast %struct.bar* %B to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.p0i8.i32(i8* %B1, i8 1, i32 24, i32 16, i1 false)
	%tmp3 = getelementptr %struct.bar, %struct.bar* %B, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp3
	%tmp4 = getelementptr %struct.bar, %struct.bar* %B, i32 0, i32 2		; <double*> [#uses=1]
	store double 1.000000e+01, double* %tmp4
	%tmp6 = getelementptr %struct.bar, %struct.bar* %B, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp7 = load i32, i32* %tmp6		; <i32> [#uses=1]
	ret i32 %tmp7
}


	%struct.f = type { i32, i32, i32, i32, i32, i32 }

define i16 @test4() nounwind {
entry:
	%A = alloca %struct.f, align 8		; <%struct.f*> [#uses=3]
	%0 = getelementptr %struct.f, %struct.f* %A, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %0, align 8
	%1 = getelementptr %struct.f, %struct.f* %A, i32 0, i32 1		; <i32*> [#uses=1]
	%2 = bitcast i32* %1 to i8*		; <i8*> [#uses=1]
	call void @llvm.memset.p0i8.i32(i8* %2, i8 2, i32 12, i32 4, i1 false)
	%3 = getelementptr %struct.f, %struct.f* %A, i32 0, i32 2		; <i32*> [#uses=1]
	%4 = load i32, i32* %3, align 8		; <i32> [#uses=1]
	%retval12 = trunc i32 %4 to i16		; <i16> [#uses=1]
	ret i16 %retval12
}
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
