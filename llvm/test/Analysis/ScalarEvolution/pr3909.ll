; RUN: opt < %s -passes=indvars -disable-output
; PR 3909


	%0 = type { i32, %1* }		; type %0
	%1 = type { i32, i8* }		; type %1

define x86_stdcallcc i32 @_Dmain(%0 %unnamed) {
entry:
	br label %whilebody

whilebody:		; preds = %endwhile5, %entry
	%i.0 = phi i64 [ 0, %entry ], [ %tmp11, %endwhile5 ]		; <i64> [#uses=1]
	%m.0 = phi i64 [ 0, %entry ], [ %tmp11, %endwhile5 ]		; <i64> [#uses=2]
	%tmp2 = mul i64 %m.0, %m.0		; <i64> [#uses=1]
	br label %whilecond3

whilecond3:		; preds = %whilebody4, %whilebody
	%j.0 = phi i64 [ %tmp2, %whilebody ], [ %tmp9, %whilebody4 ]		; <i64> [#uses=2]
	%tmp7 = icmp ne i64 %j.0, 0		; <i1> [#uses=1]
	br i1 %tmp7, label %whilebody4, label %endwhile5

whilebody4:		; preds = %whilecond3
	%tmp9 = add i64 %j.0, 1		; <i64> [#uses=1]
	br label %whilecond3

endwhile5:		; preds = %whilecond3
	%tmp11 = add i64 %i.0, 1		; <i64> [#uses=2]
	br label %whilebody
}
