; New testcase, this contains a bunch of simple instructions that should be
; handled by a code generator.

; RUN: llvm-as < %s | llc

define i32 @add(i32 %A, i32 %B) {
	%R = add i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @sub(i32 %A, i32 %B) {
	%R = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @mul(i32 %A, i32 %B) {
	%R = mul i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @sdiv(i32 %A, i32 %B) {
	%R = sdiv i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @udiv(i32 %A, i32 %B) {
	%R = udiv i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @srem(i32 %A, i32 %B) {
	%R = srem i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @urem(i32 %A, i32 %B) {
	%R = urem i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @and(i32 %A, i32 %B) {
	%R = and i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @or(i32 %A, i32 %B) {
	%R = or i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @xor(i32 %A, i32 %B) {
	%R = xor i32 %A, %B		; <i32> [#uses=1]
	ret i32 %R
}
