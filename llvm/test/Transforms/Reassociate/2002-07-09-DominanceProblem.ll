; The reassociate pass is not preserving dominance properties correctly
;
; RUN: llvm-as < %s | opt -reassociate

define i32 @compute_dist(i32 %i, i32 %j) {
	%reg119 = sub i32 %j, %i		; <i32> [#uses=1]
	ret i32 %reg119
}


