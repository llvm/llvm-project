; RUN: llvm-as < %s | opt -gvn -dse | llvm-dis | grep {call.*memcpy.*cell} | count 2
; PR2099

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"
	%struct.s = type { [11 x i8], i32 }
@.str = internal constant [11 x i8] c"0123456789\00"		; <[11 x i8]*> [#uses=1]
@cell = weak global %struct.s zeroinitializer		; <%struct.s*> [#uses=2]

declare i32 @check(%struct.s* byval  %p) nounwind

declare i32 @strcmp(i8*, i8*) nounwind readonly 

define i32 @main() noreturn nounwind  {
entry:
	%p = alloca %struct.s, align 8		; <%struct.s*> [#uses=2]
	store i32 99, i32* getelementptr (%struct.s* @cell, i32 0, i32 1), align 4
	call void @llvm.memcpy.i32( i8* getelementptr (%struct.s* @cell, i32 0, i32 0, i32 0), i8* getelementptr ([11 x i8]* @.str, i32 0, i32 0), i32 11, i32 1 )
	%tmp = getelementptr %struct.s* %p, i32 0, i32 0, i32 0		; <i8*> [#uses=2]
	call void @llvm.memcpy.i64( i8* %tmp, i8* getelementptr (%struct.s* @cell, i32 0, i32 0, i32 0), i64 16, i32 8 )
	%tmp1.i = getelementptr %struct.s* %p, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp2.i = load i32* %tmp1.i, align 4		; <i32> [#uses=1]
	%tmp3.i = icmp eq i32 %tmp2.i, 99		; <i1> [#uses=1]
	br i1 %tmp3.i, label %bb5.i, label %bb

bb5.i:		; preds = %entry
	%tmp91.i = call i32 @strcmp( i8* %tmp, i8* getelementptr ([11 x i8]* @.str, i32 0, i32 0) ) nounwind readonly 		; <i32> [#uses=1]
	%tmp53 = icmp eq i32 %tmp91.i, 0		; <i1> [#uses=1]
	br i1 %tmp53, label %bb7, label %bb

bb:		; preds = %bb5.i, %entry
	call void @abort( ) noreturn nounwind 
	unreachable

bb7:		; preds = %bb5.i
	call void @exit( i32 0 ) noreturn nounwind 
	unreachable
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 

declare void @abort() noreturn nounwind 

declare void @exit(i32) noreturn nounwind 

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32) nounwind 
