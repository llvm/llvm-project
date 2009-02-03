; RUN: llvm-as < %s | opt -reassociate -disable-output

define void @foo() {
	%tmp162 = sub <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>> [#uses=1]
	%tmp164 = mul <4 x float> zeroinitializer, %tmp162		; <<4 x float>> [#uses=0]
	ret void
}

