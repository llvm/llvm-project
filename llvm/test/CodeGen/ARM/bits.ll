; RUN: llc < %s -march=arm | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK: f1
; CHECK: and r0, r1, r0
	%tmp2 = and i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK: f2
; CHECK: orr r0, r1, r0
	%tmp2 = or i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK: f3
; CHECK: eor r0, r1, r0
	%tmp2 = xor i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK: f4
; CHECK: lsl
	%tmp3 = shl i32 %a, %b		; <i32> [#uses=1]
	ret i32 %tmp3
}

define i32 @f5(i32 %a, i32 %b) {
entry:
; CHECK: f5
; CHECK: asr
	%tmp3 = ashr i32 %a, %b		; <i32> [#uses=1]
	ret i32 %tmp3
}
