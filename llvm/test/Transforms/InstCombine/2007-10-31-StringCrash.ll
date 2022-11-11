; RUN: opt < %s -passes=instcombine -disable-output
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

declare void @__darwin_gcc3_preregister_frame_info()

define void @_start(i32 %argc, ptr %argv, ptr %envp) {
entry:
	%tmp2 = load i32, ptr @__darwin_gcc3_preregister_frame_info, align 4		; <i32> [#uses=1]
	%tmp3 = icmp ne i32 %tmp2, 0		; <i1> [#uses=1]
	%tmp34 = zext i1 %tmp3 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp34, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %return

cond_true:		; preds = %entry
	ret void

return:		; preds = %entry
	ret void
}
