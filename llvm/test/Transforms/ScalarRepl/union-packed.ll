; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | \
; RUN:   not grep alloca
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | \
; RUN:   grep bitcast

define <4 x i32> @test(<4 x float> %X) {
	%X_addr = alloca <4 x float>		; <<4 x float>*> [#uses=2]
	store <4 x float> %X, <4 x float>* %X_addr
	%X_addr.upgrd.1 = bitcast <4 x float>* %X_addr to <4 x i32>*		; <<4 x i32>*> [#uses=1]
	%tmp = load <4 x i32>* %X_addr.upgrd.1		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp
}

