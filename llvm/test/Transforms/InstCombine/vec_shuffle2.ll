; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep undef | count 1
; END.

; Test fold of two shuffles where the first shuffle vectors inputs are a
; different length then the second.

define void @test_cl(<4 x i8> addrspace(1)* %dest, <16 x i8> addrspace(1)* %old) nounwind {
entry:
	%arrayidx = getelementptr <4 x i8> addrspace(1)* %dest, i32 0		; <<4 x i8> addrspace(1)*> [#uses=1]
	%arrayidx5 = getelementptr <16 x i8> addrspace(1)* %old, i32 0		; <<16 x i8> addrspace(1)*> [#uses=1]
	%tmp6 = load <16 x i8> addrspace(1)* %arrayidx5		; <<16 x i8>> [#uses=1]
	%tmp7 = shufflevector <16 x i8> %tmp6, <16 x i8> undef, <4 x i32> < i32 13, i32 9, i32 4, i32 13 >		; <<4 x i8>> [#uses=1]
	%tmp9 = shufflevector <4 x i8> %tmp7, <4 x i8> undef, <4 x i32> < i32 3, i32 1, i32 2, i32 0 >		; <<4 x i8>> [#uses=1]
	store <4 x i8> %tmp9, <4 x i8> addrspace(1)* %arrayidx
	ret void

return:		; preds = %entry
	ret void
}