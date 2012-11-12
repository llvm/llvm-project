; RUN: opt < %s -reassociate -disable-output

define void @foo() {
	%tmp162 = fsub <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>> [#uses=1]
	%tmp164 = fmul <4 x float> zeroinitializer, %tmp162		; <<4 x float>> [#uses=0]
	ret void
}

