; PR1271
; RUN: opt < %s -instcombine -S | grep and
define i1 @test(i32 %tmp13) {
entry:
	%tmp14 = shl i32 %tmp13, 12		; <i32> [#uses=1]
	%tmp15 = lshr i32 %tmp14, 12		; <i32> [#uses=1]
	%res = icmp ne i32 %tmp15, 0		; <i1>:3 [#uses=1]
        ret i1 %res
}
