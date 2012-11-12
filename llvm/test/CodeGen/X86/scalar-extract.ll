; RUN: llc < %s -march=x86 -mattr=+mmx -o %t
; RUN: not grep movq  %t

; Check that widening doesn't introduce a mmx register in this case when
; a simple load/store would suffice.

define void @foo(<2 x i16>* %A, <2 x i16>* %B) {
entry:
	%tmp1 = load <2 x i16>* %A		; <<2 x i16>> [#uses=1]
	store <2 x i16> %tmp1, <2 x i16>* %B
	ret void
}

