; RUN: opt < %s -passes=instcombine | llvm-dis
; PR3667
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @_ada_c32001b(i32 %tmp5, ptr %src) {
entry:
	%max289 = select i1 false, i32 %tmp5, i32 0		; <i32> [#uses=1]
	%tmp6 = mul i32 %max289, 4		; <i32> [#uses=1]
	%tmp7 = alloca i8, i32 0		; <ptr> [#uses=1]
	%tmp11 = load i32, ptr %src, align 1		; <i32> [#uses=1]
	%tmp12 = icmp eq i32 %tmp11, 3		; <i1> [#uses=1]
	%tmp13 = zext i1 %tmp12 to i8		; <i8> [#uses=1]
	%tmp14 = ashr i32 %tmp6, 2		; <i32> [#uses=1]
	%tmp16 = mul i32 %tmp14, 4		; <i32> [#uses=1]
	%tmp17 = mul i32 1, %tmp16		; <i32> [#uses=1]
	%tmp18 = getelementptr i8, ptr %tmp7, i32 %tmp17		; <ptr> [#uses=1]
	%tmp21 = getelementptr i32, ptr %tmp18, i32 0		; <ptr> [#uses=1]
	%tmp22 = load i32, ptr %tmp21, align 1		; <i32> [#uses=1]
	%tmp23 = icmp eq i32 %tmp22, 4		; <i1> [#uses=1]
	%tmp24 = zext i1 %tmp23 to i8		; <i8> [#uses=1]
	%toBool709 = icmp ne i8 %tmp13, 0		; <i1> [#uses=1]
	%toBool710 = icmp ne i8 %tmp24, 0		; <i1> [#uses=1]
	%tmp25 = and i1 %toBool709, %toBool710		; <i1> [#uses=1]
	%tmp26 = zext i1 %tmp25 to i8		; <i8> [#uses=1]
	%toBool711 = icmp ne i8 %tmp26, 0		; <i1> [#uses=1]
	br i1 %toBool711, label %a, label %b

a:		; preds = %entry
	ret void

b:		; preds = %entry
	ret void
}
