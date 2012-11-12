; RUN: llc < %s -march=thumb -mattr=+thumb2,+t2xtpk | FileCheck %s

; CHECK: test1
; CHECK: pkhbt   r0, r0, r1, lsl #16
define i32 @test1(i32 %X, i32 %Y) {
	%tmp1 = and i32 %X, 65535		; <i32> [#uses=1]
	%tmp4 = shl i32 %Y, 16		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp4, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp5
}

; CHECK: test1a
; CHECK: pkhbt   r0, r0, r1, lsl #16
define i32 @test1a(i32 %X, i32 %Y) {
	%tmp19 = and i32 %X, 65535		; <i32> [#uses=1]
	%tmp37 = shl i32 %Y, 16		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp37, %tmp19		; <i32> [#uses=1]
	ret i32 %tmp5
}

; CHECK: test2
; CHECK: pkhbt   r0, r0, r1, lsl #12
define i32 @test2(i32 %X, i32 %Y) {
	%tmp1 = and i32 %X, 65535		; <i32> [#uses=1]
	%tmp3 = shl i32 %Y, 12		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp3, -65536		; <i32> [#uses=1]
	%tmp57 = or i32 %tmp4, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp57
}

; CHECK: test3
; CHECK: pkhbt   r0, r0, r1, lsl #18
define i32 @test3(i32 %X, i32 %Y) {
	%tmp19 = and i32 %X, 65535		; <i32> [#uses=1]
	%tmp37 = shl i32 %Y, 18		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp37, %tmp19		; <i32> [#uses=1]
	ret i32 %tmp5
}

; CHECK: test4
; CHECK: pkhbt   r0, r0, r1
define i32 @test4(i32 %X, i32 %Y) {
	%tmp1 = and i32 %X, 65535		; <i32> [#uses=1]
	%tmp3 = and i32 %Y, -65536		; <i32> [#uses=1]
	%tmp46 = or i32 %tmp3, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp46
}

; CHECK: test5
; CHECK: pkhtb   r0, r0, r1, asr #16
define i32 @test5(i32 %X, i32 %Y) {
	%tmp17 = and i32 %X, -65536		; <i32> [#uses=1]
	%tmp2 = bitcast i32 %Y to i32		; <i32> [#uses=1]
	%tmp4 = lshr i32 %tmp2, 16		; <i32> [#uses=2]
	%tmp5 = or i32 %tmp4, %tmp17		; <i32> [#uses=1]
	ret i32 %tmp5
}

; CHECK: test5a
; CHECK: pkhtb   r0, r0, r1, asr #16
define i32 @test5a(i32 %X, i32 %Y) {
	%tmp110 = and i32 %X, -65536		; <i32> [#uses=1]
	%tmp37 = lshr i32 %Y, 16		; <i32> [#uses=1]
	%tmp39 = bitcast i32 %tmp37 to i32		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp39, %tmp110		; <i32> [#uses=1]
	ret i32 %tmp5
}

; CHECK: test6
; CHECK: pkhtb   r0, r0, r1, asr #12
define i32 @test6(i32 %X, i32 %Y) {
	%tmp1 = and i32 %X, -65536		; <i32> [#uses=1]
	%tmp37 = lshr i32 %Y, 12		; <i32> [#uses=1]
	%tmp38 = bitcast i32 %tmp37 to i32		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp38, 65535		; <i32> [#uses=1]
	%tmp59 = or i32 %tmp4, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp59
}

; CHECK: test7
; CHECK: pkhtb   r0, r0, r1, asr #18
define i32 @test7(i32 %X, i32 %Y) {
	%tmp1 = and i32 %X, -65536		; <i32> [#uses=1]
	%tmp3 = ashr i32 %Y, 18		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp3, 65535		; <i32> [#uses=1]
	%tmp57 = or i32 %tmp4, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp57
}

; CHECK: test8
; CHECK: pkhtb   r0, r0, r1, asr #22
define i32 @test8(i32 %X, i32 %Y) {
	%tmp1 = and i32 %X, -65536
	%tmp3 = lshr i32 %Y, 22
	%tmp57 = or i32 %tmp3, %tmp1
	ret i32 %tmp57
}
