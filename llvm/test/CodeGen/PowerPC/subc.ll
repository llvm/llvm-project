; All of these should be codegen'd without loading immediates
; RUN: llvm-as < %s | llc -march=ppc32 -o %t -f
; RUN: grep subfc %t | count 1
; RUN: grep subfe %t | count 1
; RUN: grep subfze %t | count 1
; RUN: grep subfme %t | count 1
; RUN: grep subfic %t | count 2

define i64 @sub_ll(i64 %a, i64 %b) {
entry:
	%tmp.2 = sub i64 %a, %b		; <i64> [#uses=1]
	ret i64 %tmp.2
}

define i64 @sub_l_5(i64 %a) {
entry:
	%tmp.1 = sub i64 5, %a		; <i64> [#uses=1]
	ret i64 %tmp.1
}

define i64 @sub_l_m5(i64 %a) {
entry:
	%tmp.1 = sub i64 -5, %a		; <i64> [#uses=1]
	ret i64 %tmp.1
}
