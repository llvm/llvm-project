; RUN: opt < %s -passes=instcombine -S | grep "store volatile"

define void @test() {
	%votf = alloca <4 x float>		; <ptr> [#uses=1]
	store volatile <4 x float> zeroinitializer, ptr %votf, align 16
	ret void
}

