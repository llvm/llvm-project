; RUN: opt < %s -aa-pipeline=basic-aa -gvn -disable-output
; PR2395

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.S291 = type <{ %union.anon, i32 }>
	%union.anon = type {  }
@a291 = external global [5 x %struct.S291]		; <[5 x %struct.S291]*> [#uses=2]

define void @test291() nounwind  {
entry:
	store i32 1138410269, i32* getelementptr ([5 x %struct.S291], [5 x %struct.S291]* @a291, i32 0, i32 2, i32 1)
	%tmp54 = load i32, i32* bitcast (%struct.S291* getelementptr ([5 x %struct.S291], [5 x %struct.S291]* @a291, i32 0, i32 2) to i32*), align 4		; <i32> [#uses=0]
	unreachable
}
