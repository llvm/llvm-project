; RUN: opt < %s -passes=instcombine -S | grep "mul i64"
; rdar://6762288

; Instcombine should not promote the mul to i96 because it is definitely
; not a legal type for the target, and we don't want a libcall.

define i96 @test(i96 %a.4, i96 %b.2) {
	%tmp1086 = trunc i96 %a.4 to i64		; <i64> [#uses=1]
	%tmp836 = trunc i96 %b.2 to i64		; <i64> [#uses=1]
	%mul185 = mul i64 %tmp1086, %tmp836		; <i64> [#uses=1]
	%tmp544 = zext i64 %mul185 to i96		; <i96> [#uses=1]
	ret i96 %tmp544
}
