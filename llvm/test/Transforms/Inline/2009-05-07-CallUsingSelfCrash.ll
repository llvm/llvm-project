; RUN: opt < %s -inline -disable-output
; PR4123
	%struct.S0 = type <{ i32 }>
	%struct.S1 = type <{ i8, i8, i8, i8, %struct.S0 }>
	%struct.S2 = type <{ %struct.S1, i32 }>

define void @func_113(%struct.S1* noalias nocapture sret %agg.result, i8 signext %p_114) noreturn nounwind {
entry:
	unreachable

for.inc:		; preds = %for.inc
	%call48 = call fastcc signext i8 @safe_sub_func_uint8_t_u_u(i8 signext %call48)		; <i8> [#uses=1]
	br label %for.inc
}

define fastcc signext i8 @safe_sub_func_uint8_t_u_u(i8 signext %_ui1) nounwind readnone {
entry:
	ret i8 %_ui1
}

