; PR1226
; RUN: opt < %s -scalarrepl -S | \
; RUN:   not grep "call void @llvm.memcpy.p0i8.p0i8.i32"
; RUN: opt < %s -scalarrepl -S | grep getelementptr
; END.

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.8.0"
	%struct.foo = type { i8, i8 }


define i32 @test1(%struct.foo* %P) {
entry:
	%L = alloca %struct.foo, align 2		; <%struct.foo*> [#uses=1]
	%L2 = getelementptr %struct.foo, %struct.foo* %L, i32 0, i32 0		; <i8*> [#uses=2]
	%tmp13 = getelementptr %struct.foo, %struct.foo* %P, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memcpy.p0i8.p0i8.i32( i8* %L2, i8* %tmp13, i32 2, i1 false)
	%tmp5 = load i8, i8* %L2		; <i8> [#uses=1]
	%tmp56 = sext i8 %tmp5 to i32		; <i32> [#uses=1]
	ret i32 %tmp56
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1)
