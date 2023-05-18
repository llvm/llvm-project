; RUN: llc < %s  -mtriple=i686-pc-linux-gnu -relocation-model=pic | FileCheck %s

define tailcc i32 @tailcallee(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
entry:
	ret i32 %a3
}

define tailcc i32 @tailcaller(i32 %in1, i32 %in2) {
entry:
	%tmp11 = tail call tailcc i32 @tailcallee( i32 %in1, i32 %in2, i32 %in1, i32 %in2 )		; <i32> [#uses=1]
	ret i32 %tmp11
; CHECK: movl tailcallee@GOT
; CHECK: jmpl
}

