; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 -disable-mmx -o %t -f
; RUN: not grep pextrw %t
; RUN: grep pinsrw %t

; Test for v8xi16 lowering where we extract the first element of the vector and
; placed it in the second element of the result.

define void @test_cl(<8 x i16> addrspace(1)* %dest, <8 x i16> addrspace(1)* %old) nounwind {
entry:
	%tmp3 = load <8 x i16> addrspace(1)* %old		; <<8 x i16>> [#uses=1]
	%tmp6 = shufflevector <8 x i16> %tmp3, <8 x i16> < i16 0, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef >, <8 x i32> < i32 8, i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef  >		; <<8 x i16>> [#uses=1]
	store <8 x i16> %tmp6, <8 x i16> addrspace(1)* %dest
	ret void
}