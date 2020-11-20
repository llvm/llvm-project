; RUN: opt < %s -basic-aa -gvn -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
	%struct.x = type { i32, i32, i32, i32 }
@g = weak global i32 0		; <i32*> [#uses=1]

define i32 @foo(%struct.x* byval(%struct.x) %a) nounwind  {
; CHECK: ret i32 1
  %tmp1 = tail call i32 (...) @bar( %struct.x* %a ) nounwind 		; <i32> [#uses=0]
  %tmp2 = getelementptr %struct.x, %struct.x* %a, i32 0, i32 0		; <i32*> [#uses=2]
  store i32 1, i32* %tmp2, align 4
  store i32 2, i32* @g, align 4
  %tmp4 = load i32, i32* %tmp2, align 4		; <i32> [#uses=1]
  ret i32 %tmp4
}

declare i32 @bar(...)

