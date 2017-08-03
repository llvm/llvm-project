; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep rotrw: | count 1
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep rotlw: | count 1
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep rotlwi: | count 1
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep rotrwi: | count 1

define i32 @rotlw(i32 %x, i32 %sh) {
entry:
	%tmp.7 = sub i32 32, %sh		; <i32> [#uses=1]
	%tmp.10 = lshr i32 %x, %tmp.7		; <i32> [#uses=2]
	%tmp.4 = shl i32 %x, %sh 		; <i32> [#uses=1]
	%tmp.12 = or i32 %tmp.10, %tmp.4		; <i32> [#uses=1]
	ret i32 %tmp.12
}

define i32 @rotrw(i32 %x, i32 %sh) {
entry:
	%tmp.3 = trunc i32 %sh to i8		; <i8> [#uses=1]
	%tmp.4 = lshr i32 %x, %sh		; <i32> [#uses=2]
	%tmp.7 = sub i32 32, %sh		; <i32> [#uses=1]
	%tmp.10 = shl i32 %x, %tmp.7    	; <i32> [#uses=1]
	%tmp.12 = or i32 %tmp.4, %tmp.10		; <i32> [#uses=1]
	ret i32 %tmp.12
}

define i32 @rotlwi(i32 %x) {
entry:
	%tmp.7 = lshr i32 %x, 27		; <i32> [#uses=2]
	%tmp.3 = shl i32 %x, 5		; <i32> [#uses=1]
	%tmp.9 = or i32 %tmp.3, %tmp.7		; <i32> [#uses=1]
	ret i32 %tmp.9
}

define i32 @rotrwi(i32 %x) {
entry:
	%tmp.3 = lshr i32 %x, 5		; <i32> [#uses=2]
	%tmp.7 = shl i32 %x, 27		; <i32> [#uses=1]
	%tmp.9 = or i32 %tmp.3, %tmp.7		; <i32> [#uses=1]
	ret i32 %tmp.9
}
