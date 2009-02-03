; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | \
; RUN:   not grep alloca
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | \
; RUN:   grep {bitcast.*float.*i32}

define i32 @test(float %X) {
	%X_addr = alloca float		; <float*> [#uses=2]
	store float %X, float* %X_addr
	%X_addr.upgrd.1 = bitcast float* %X_addr to i32*		; <i32*> [#uses=1]
	%tmp = load i32* %X_addr.upgrd.1		; <i32> [#uses=1]
	ret i32 %tmp
}

