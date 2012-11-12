; RUN: opt < %s -instcombine -S | grep "ugt.*, 1"

define i1 @test(i32 %tmp1030) {
	%tmp1037 = icmp ne i32 %tmp1030, 40		; <i1> [#uses=1]
	%tmp1039 = icmp ne i32 %tmp1030, 41		; <i1> [#uses=1]
	%tmp1042 = and i1 %tmp1037, %tmp1039		; <i1> [#uses=1]
	ret i1 %tmp1042
}
