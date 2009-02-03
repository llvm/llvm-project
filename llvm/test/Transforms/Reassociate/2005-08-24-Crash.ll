; RUN: llvm-as < %s | opt -reassociate -disable-output

define void @test(i32 %a, i32 %b, i32 %c, i32 %d) {
	%tmp.2 = xor i32 %a, %b		; <i32> [#uses=1]
	%tmp.5 = xor i32 %c, %d		; <i32> [#uses=1]
	%tmp.6 = xor i32 %tmp.2, %tmp.5		; <i32> [#uses=1]
	%tmp.9 = xor i32 %c, %a		; <i32> [#uses=1]
	%tmp.12 = xor i32 %b, %d		; <i32> [#uses=1]
	%tmp.13 = xor i32 %tmp.9, %tmp.12		; <i32> [#uses=1]
	%tmp.16 = xor i32 %tmp.6, %tmp.13		; <i32> [#uses=0]
	ret void
}

