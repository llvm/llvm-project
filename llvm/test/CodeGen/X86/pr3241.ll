; RUN: llc < %s -march=x86
; PR3241

@g_620 = external global i32

define void @func_18(i32 %p_21) nounwind {
entry:
	%t0 = call i32 @func_31(i32 %p_21) nounwind
	%t1 = call i32 @safe_add_macro_uint32_t_u_u() nounwind
	%t2 = icmp sgt i32 %t1, 0
	%t3 = zext i1 %t2 to i32
	%t4 = load i32, i32* @g_620, align 4
	%t5 = icmp eq i32 %t3, %t4
	%t6 = xor i32 %p_21, 1
	%t7 = call i32 @func_55(i32 %t6) nounwind
	br i1 %t5, label %return, label %bb

bb:
	unreachable

return:
	unreachable
}

declare i32 @func_31(i32)

declare i32 @safe_add_macro_uint32_t_u_u()

declare i32 @func_55(i32)
