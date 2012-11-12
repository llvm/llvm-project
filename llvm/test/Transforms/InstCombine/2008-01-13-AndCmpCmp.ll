; RUN: opt < %s -instcombine -S | grep and
; PR1907

define i1 @test(i32 %c84.17) {
	%tmp2696 = icmp ne i32 %c84.17, 34		; <i1> [#uses=2]
 	%tmp2699 = icmp sgt i32 %c84.17, -1		; <i1> [#uses=1]
 	%tmp2703 = and i1 %tmp2696, %tmp2699		; <i1> [#uses=1]
	ret i1 %tmp2703
}
