; RUN: llvm-as < %s | llc -march=x86 | grep movsw
; PR2139

declare void @abort()

define i32 @main() {
entry:
	%tmp73 = tail call i1 @return_false()		; <i8> [#uses=1]
	%g.0 = select i1 %tmp73, i16 0, i16 -480		; <i16> [#uses=2]
	%tmp7778 = sext i16 %g.0 to i32		; <i32> [#uses=1]
	%tmp80 = shl i32 %tmp7778, 3		; <i32> [#uses=2]
	%tmp87 = icmp sgt i32 %tmp80, 32767		; <i1> [#uses=1]
	br i1 %tmp87, label %bb90, label %bb91
bb90:		; preds = %bb84, %bb72
	tail call void @abort()
	unreachable
bb91:		; preds = %bb84
	ret i32 0
}

define i1 @return_false() {
	ret i1 0
}
