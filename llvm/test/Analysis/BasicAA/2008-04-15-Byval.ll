; RUN: opt < %s -O3 -S | FileCheck %s
; ModuleID = 'small2.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
	%struct.x = type { [4 x i32] }

define void @foo(%struct.x* byval(%struct.x) align 4  %X) nounwind  {
; CHECK: store i32 2, i32* %tmp1
entry:
	%tmp = getelementptr %struct.x, %struct.x* %X, i32 0, i32 0		; <[4 x i32]*> [#uses=1]
	%tmp1 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 3		; <i32*> [#uses=1]
	store i32 2, i32* %tmp1, align 4
	%tmp2 = call i32 (...) @bar(%struct.x* byval(%struct.x) align 4  %X ) nounwind 		; <i32> [#uses=0]
	br label %return
return:		; preds = %entry
	ret void
}

declare i32 @bar(...)
