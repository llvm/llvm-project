; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {s = alloca .struct.x}
; PR2423
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
	%struct.x = type { [1 x i32], i32, i32 }

define i32 @b() nounwind {
entry:
	%s = alloca %struct.x		; <%struct.x*> [#uses=2]
	%r = alloca %struct.x		; <%struct.x*> [#uses=2]
	call i32 @a( %struct.x* %s ) nounwind		; <i32>:0 [#uses=0]
	%r1 = bitcast %struct.x* %r to i8*		; <i8*> [#uses=1]
	%s2 = bitcast %struct.x* %s to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %r1, i8* %s2, i32 12, i32 8 )
	getelementptr %struct.x* %r, i32 0, i32 0, i32 1		; <i32*>:1 [#uses=1]
	load i32* %1, align 4		; <i32>:2 [#uses=1]
	ret i32 %2
}

declare i32 @a(%struct.x*)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind
