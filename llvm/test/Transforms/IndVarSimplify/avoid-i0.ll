; RUN: opt < %s -passes=indvars
; PR4052
; PR4054

; Don't treat an and with 0 as a mask (trunc+zext).

define i32 @int80(i8 signext %p_71) nounwind {
entry:
	br label %bb

bb:		; preds = %bb6, %entry
	%p_71_addr.0 = phi i8 [ %p_71, %entry ], [ %0, %bb6 ]		; <i8> [#uses=0]
	br i1 undef, label %bb4, label %bb1

bb1:		; preds = %bb
	ret i32 0

bb4:		; preds = %bb4, %bb
	br i1 undef, label %bb6, label %bb4

bb6:		; preds = %bb4
	%0 = and i8 0, 0		; <i8> [#uses=1]
	br label %bb
}

@x = common global i32 0		; <ptr> [#uses=1]

define signext i8 @safe_sub_func_int32_t_s_s(i32 %_si1, i8 signext %_si2) nounwind {
entry:
	%_si1_addr = alloca i32		; <ptr> [#uses=3]
	%_si2_addr = alloca i8		; <ptr> [#uses=3]
	%retval = alloca i32		; <ptr> [#uses=2]
	%0 = alloca i32		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %_si1, ptr %_si1_addr
	store i8 %_si2, ptr %_si2_addr
	%1 = load i8, ptr %_si2_addr, align 1		; <i8> [#uses=1]
	%2 = sext i8 %1 to i32		; <i32> [#uses=1]
	%3 = load i32, ptr %_si1_addr, align 4		; <i32> [#uses=1]
	%4 = xor i32 %2, %3		; <i32> [#uses=1]
	%5 = load i8, ptr %_si2_addr, align 1		; <i8> [#uses=1]
	%6 = sext i8 %5 to i32		; <i32> [#uses=1]
	%7 = sub i32 7, %6		; <i32> [#uses=1]
	%8 = load i32, ptr %_si1_addr, align 4		; <i32> [#uses=1]
	%9 = shl i32 %8, %7		; <i32> [#uses=1]
	%10 = and i32 %4, %9		; <i32> [#uses=1]
	%11 = icmp slt i32 %10, 0		; <i1> [#uses=1]
	%12 = zext i1 %11 to i32		; <i32> [#uses=1]
	store i32 %12, ptr %0, align 4
	%13 = load i32, ptr %0, align 4		; <i32> [#uses=1]
	store i32 %13, ptr %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32, ptr %retval		; <i32> [#uses=1]
	%retval12 = trunc i32 %retval1 to i8		; <i8> [#uses=1]
	ret i8 %retval12
}

define i32 @safe_sub_func_uint64_t_u_u(i32 %_ui1, i32 %_ui2) nounwind {
entry:
	%_ui1_addr = alloca i32		; <ptr> [#uses=2]
	%_ui2_addr = alloca i32		; <ptr> [#uses=1]
	%retval = alloca i32		; <ptr> [#uses=2]
	%0 = alloca i32		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %_ui1, ptr %_ui1_addr
	store i32 %_ui2, ptr %_ui2_addr
	%1 = load i32, ptr %_ui1_addr, align 4		; <i32> [#uses=1]
	%2 = sub i32 %1, 1		; <i32> [#uses=1]
	store i32 %2, ptr %0, align 4
	%3 = load i32, ptr %0, align 4		; <i32> [#uses=1]
	store i32 %3, ptr %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval1
}

define void @int87(i8 signext %p_48, i8 signext %p_49) nounwind {
entry:
	%p_48_addr = alloca i8		; <ptr> [#uses=1]
	%p_49_addr = alloca i8		; <ptr> [#uses=1]
	%l_52 = alloca i32		; <ptr> [#uses=7]
	%vol.0 = alloca i32		; <ptr> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i8 %p_48, ptr %p_48_addr
	store i8 %p_49, ptr %p_49_addr
	br label %bb4

bb:		; preds = %bb4
	%0 = load volatile i32, ptr @x, align 4		; <i32> [#uses=1]
	store i32 %0, ptr %vol.0, align 4
	store i32 0, ptr %l_52, align 4
	br label %bb2

bb1:		; preds = %bb2
	%1 = load i32, ptr %l_52, align 4		; <i32> [#uses=1]
	%2 = call i32 @safe_sub_func_uint64_t_u_u(i32 %1, i32 1) nounwind		; <i32> [#uses=1]
	store i32 %2, ptr %l_52, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	%3 = load i32, ptr %l_52, align 4		; <i32> [#uses=1]
	%4 = icmp eq i32 %3, 0		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb3

bb3:		; preds = %bb2
	%5 = load i32, ptr %l_52, align 4		; <i32> [#uses=1]
	%6 = call signext i8 @safe_sub_func_int32_t_s_s(i32 %5, i8 signext 1) nounwind		; <i8> [#uses=1]
	%7 = sext i8 %6 to i32		; <i32> [#uses=1]
	store i32 %7, ptr %l_52, align 4
	br label %bb4

bb4:		; preds = %bb3, %entry
	%8 = load i32, ptr %l_52, align 4		; <i32> [#uses=1]
	%9 = icmp ne i32 %8, 0		; <i1> [#uses=1]
	br i1 %9, label %bb, label %bb5

bb5:		; preds = %bb4
	br label %return

return:		; preds = %bb5
	ret void
}
