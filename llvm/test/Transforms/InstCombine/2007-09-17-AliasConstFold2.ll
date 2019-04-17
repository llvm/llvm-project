; RUN: opt < %s -instcombine -S | grep icmp
; PR1678

@A = weak alias void (), void ()* @B		; <void ()*> [#uses=1]

define weak void @B() {
       ret void
}

define i32 @active() {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp1 = icmp ne void ()* @A, null		; <i1> [#uses=1]
	%tmp12 = zext i1 %tmp1 to i32		; <i32> [#uses=1]
	ret i32 %tmp12
}
