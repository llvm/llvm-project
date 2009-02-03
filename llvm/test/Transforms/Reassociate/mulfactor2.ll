; This should turn into one multiply and one add.

; RUN: llvm-as < %s | \
; RUN:   opt -instcombine -reassociate -instcombine | llvm-dis > %t 
; RUN: grep mul %t | count 1
; RUN: grep add %t | count 1

define i32 @main(i32 %t) {
	%tmp.3 = mul i32 %t, 12		; <i32> [#uses=1]
	%tmp.4 = add i32 %tmp.3, 5		; <i32> [#uses=1]
	%tmp.6 = mul i32 %t, 6		; <i32> [#uses=1]
	%tmp.8 = mul i32 %tmp.4, 3		; <i32> [#uses=1]
	%tmp.9 = add i32 %tmp.8, %tmp.6		; <i32> [#uses=1]
	ret i32 %tmp.9
}

