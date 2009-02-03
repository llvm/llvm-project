; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse41 -o %t -f

; tests variable insert and extract of a 4 x i32

define <4 x i32> @var_insert(<4 x i32> %x, i32 %val, i32 %idx) nounwind  {
entry:
	%tmp3 = insertelement <4 x i32> %x, i32 %val, i32 %idx		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp3
}

define i32 @var_extract(<4 x i32> %x, i32 %idx) nounwind  {
entry:
	%tmp3 = extractelement <4 x i32> %x, i32 %idx		; <<i32>> [#uses=1]
	ret i32 %tmp3
}
