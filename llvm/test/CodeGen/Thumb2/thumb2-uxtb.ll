; RUN: llc < %s -march=thumb -mcpu=cortex-a8 | FileCheck %s -check-prefix=ARMv7A
; RUN: llc < %s -march=thumb -mcpu=cortex-m3 | FileCheck %s -check-prefix=ARMv7M

define i32 @test1(i32 %x) {
; ARMv7A: test1
; ARMv7A: uxtb16 r0, r0

; ARMv7M: test1
; ARMv7M: bic r0, r0, #-16711936
	%tmp1 = and i32 %x, 16711935		; <i32> [#uses=1]
	ret i32 %tmp1
}

; PR7503
define i32 @test2(i32 %x) {
; ARMv7A: test2
; ARMv7A: uxtb16  r0, r0, ror #8

; ARMv7M: test2
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, lsr #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test3(i32 %x) {
; ARMv7A: test3
; ARMv7A: uxtb16  r0, r0, ror #8

; ARMv7M: test3
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, lsr #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test4(i32 %x) {
; ARMv7A: test4
; ARMv7A: uxtb16  r0, r0, ror #8

; ARMv7M: test4
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, lsr #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp6 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test5(i32 %x) {
; ARMv7A: test5
; ARMv7A: uxtb16  r0, r0, ror #8

; ARMv7M: test5
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, lsr #8
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test6(i32 %x) {
; ARMv7A: test6
; ARMv7A: uxtb16  r0, r0, ror #16

; ARMv7M: test6
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, ror #16
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test7(i32 %x) {
; ARMv7A: test7
; ARMv7A: uxtb16  r0, r0, ror #16

; ARMv7M: test7
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, ror #16
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test8(i32 %x) {
; ARMv7A: test8
; ARMv7A: uxtb16  r0, r0, ror #24

; ARMv7M: test8
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, ror #24
	%tmp1 = shl i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711680		; <i32> [#uses=1]
	%tmp5 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test9(i32 %x) {
; ARMv7A: test9
; ARMv7A: uxtb16  r0, r0, ror #24

; ARMv7M: test9
; ARMv7M: mov.w r1, #16711935
; ARMv7M: and.w r0, r1, r0, ror #24
	%tmp1 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 8		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp5, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test10(i32 %p0) {
; ARMv7A: test10
; ARMv7A: mov.w r1, #16253176
; ARMv7A: and.w r0, r1, r0, lsr #7
; ARMv7A: lsrs  r1, r0, #5
; ARMv7A: uxtb16  r1, r1
; ARMv7A: orrs r0, r1

; ARMv7M: test10
; ARMv7M: mov.w r1, #16253176
; ARMv7M: and.w r0, r1, r0, lsr #7
; ARMv7M: mov.w r1, #458759
; ARMv7M: and.w r1, r1, r0, lsr #5
; ARMv7M: orrs r0, r1
	%tmp1 = lshr i32 %p0, 7		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16253176		; <i32> [#uses=2]
	%tmp4 = lshr i32 %tmp2, 5		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 458759		; <i32> [#uses=1]
	%tmp7 = or i32 %tmp5, %tmp2		; <i32> [#uses=1]
	ret i32 %tmp7
}
