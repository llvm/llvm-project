; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
; CHECK-LABEL: @test1(
; CHECK: ret i32 %A
        %B = shl i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i8 %A) {
; CHECK-LABEL: @test2(
; CHECK: ret i32 0
        %shift.upgrd.1 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = shl i32 0, %shift.upgrd.1          ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test3(i32 %A) {
; CHECK-LABEL: @test3(
; CHECK: ret i32 %A
        %B = ashr i32 %A, 0             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test4(i8 %A) {
; CHECK-LABEL: @test4(
; CHECK: ret i32 0
        %shift.upgrd.2 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = ashr i32 0, %shift.upgrd.2         ; <i32> [#uses=1]
        ret i32 %B
}


define i32 @test5(i32 %A) {
; CHECK-LABEL: @test5(
; CHECK: ret i32 undef
        %B = lshr i32 %A, 32  ;; shift all bits out
        ret i32 %B
}

define <4 x i32> @test5_splat_vector(<4 x i32> %A) {
; CHECK-LABEL: @test5_splat_vector(
; CHECK: ret <4 x i32> undef
  %B = lshr <4 x i32> %A, <i32 32, i32 32, i32 32, i32 32>     ;; shift all bits out
  ret <4 x i32> %B
}

define <4 x i32> @test5_zero_vector(<4 x i32> %A) {
; CHECK-LABEL: @test5_zero_vector(
; CHECK-NEXT: ret <4 x i32> %A
  %B = lshr <4 x i32> %A, zeroinitializer
  ret <4 x i32> %B
}

define <4 x i32> @test5_non_splat_vector(<4 x i32> %A) {
; CHECK-LABEL: @test5_non_splat_vector(
; CHECK-NOT: ret <4 x i32> undef
  %B = lshr <4 x i32> %A, <i32 32, i32 1, i32 2, i32 3>
  ret <4 x i32> %B
}

define i32 @test5a(i32 %A) {
; CHECK-LABEL: @test5a(
; CHECK: ret i32 undef
        %B = shl i32 %A, 32     ;; shift all bits out
        ret i32 %B
}

define <4 x i32> @test5a_splat_vector(<4 x i32> %A) {
; CHECK-LABEL: @test5a_splat_vector(
; CHECK: ret <4 x i32> undef
  %B = shl <4 x i32> %A, <i32 32, i32 32, i32 32, i32 32>     ;; shift all bits out
  ret <4 x i32> %B
}

define <4 x i32> @test5a_non_splat_vector(<4 x i32> %A) {
; CHECK-LABEL: @test5a_non_splat_vector(
; CHECK-NOT: ret <4 x i32> undef
  %B = shl <4 x i32> %A, <i32 32, i32 1, i32 2, i32 3>
  ret <4 x i32> %B
}

define i32 @test5b() {
; CHECK-LABEL: @test5b(
; CHECK: ret i32 0
        %B = ashr i32 undef, 2  ;; top two bits must be equal, so not undef
        ret i32 %B
}

define i32 @test5b2(i32 %A) {
; CHECK-LABEL: @test5b2(
; CHECK: ret i32 0
        %B = ashr i32 undef, %A  ;; top %A bits must be equal, so not undef
        ret i32 %B
}

define i32 @test6(i32 %A) {
; CHECK-LABEL: @test6(
; CHECK-NEXT: mul i32 %A, 6
; CHECK-NEXT: ret i32
        %B = shl i32 %A, 1      ;; convert to an mul instruction
        %C = mul i32 %B, 3
        ret i32 %C
}

define i32 @test6a(i32 %A) {
; CHECK-LABEL: @test6a(
; CHECK-NEXT: mul i32 %A, 6
; CHECK-NEXT: ret i32
        %B = mul i32 %A, 3
        %C = shl i32 %B, 1      ;; convert to an mul instruction
        ret i32 %C
}

define i32 @test7(i8 %A) {
; CHECK-LABEL: @test7(
; CHECK-NEXT: ret i32 -1
        %shift.upgrd.3 = zext i8 %A to i32
        %B = ashr i32 -1, %shift.upgrd.3  ;; Always equal to -1
        ret i32 %B
}

;; (A << 5) << 3 === A << 8 == 0
define i8 @test8(i8 %A) {
; CHECK-LABEL: @test8(
; CHECK: ret i8 0
        %B = shl i8 %A, 5               ; <i8> [#uses=1]
        %C = shl i8 %B, 3               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A << 7) >> 7 === A & 1
define i8 @test9(i8 %A) {
; CHECK-LABEL: @test9(
; CHECK-NEXT: and i8 %A, 1
; CHECK-NEXT: ret i8
        %B = shl i8 %A, 7               ; <i8> [#uses=1]
        %C = lshr i8 %B, 7              ; <i8> [#uses=1]
        ret i8 %C
}

;; This transformation is deferred to DAGCombine:
;; (A >> 7) << 7 === A & 128
;; The shl may be valuable to scalar evolution.
define i8 @test10(i8 %A) {
; CHECK-LABEL: @test10(
; CHECK-NEXT: and i8 %A, -128
; CHECK-NEXT: ret i8
        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        %C = shl i8 %B, 7               ; <i8> [#uses=1]
        ret i8 %C
}

;; Allow the simplification when the lshr shift is exact.
define i8 @test10a(i8 %A) {
; CHECK-LABEL: @test10a(
; CHECK-NEXT: ret i8 %A
        %B = lshr exact i8 %A, 7
        %C = shl i8 %B, 7
        ret i8 %C
}

;; This transformation is deferred to DAGCombine:
;; (A >> 3) << 4 === (A & 0x1F) << 1
;; The shl may be valuable to scalar evolution.
define i8 @test11(i8 %A) {
; CHECK-LABEL: @test11(
; CHECK: shl i8
; CHECK-NEXT: ret i8
        %a = mul i8 %A, 3               ; <i8> [#uses=1]
        %B = lshr i8 %a, 3              ; <i8> [#uses=1]
        %C = shl i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

;; Allow the simplification in InstCombine when the lshr shift is exact.
define i8 @test11a(i8 %A) {
; CHECK-LABEL: @test11a(
; CHECK-NEXT: mul i8 %A, 6
; CHECK-NEXT: ret i8
        %a = mul i8 %A, 3
        %B = lshr exact i8 %a, 3
        %C = shl i8 %B, 4
        ret i8 %C
}

;; This is deferred to DAGCombine unless %B is single-use.
;; (A >> 8) << 8 === A & -256
define i32 @test12(i32 %A) {
; CHECK-LABEL: @test12(
; CHECK-NEXT: and i32 %A, -256
; CHECK-NEXT: ret i32
        %B = ashr i32 %A, 8             ; <i32> [#uses=1]
        %C = shl i32 %B, 8              ; <i32> [#uses=1]
        ret i32 %C
}

;; This transformation is deferred to DAGCombine:
;; (A >> 3) << 4 === (A & -8) * 2
;; The shl may be valuable to scalar evolution.
define i8 @test13(i8 %A) {
; CHECK-LABEL: @test13(
; CHECK: shl i8
; CHECK-NEXT: ret i8
        %a = mul i8 %A, 3               ; <i8> [#uses=1]
        %B = ashr i8 %a, 3              ; <i8> [#uses=1]
        %C = shl i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

define i8 @test13a(i8 %A) {
; CHECK-LABEL: @test13a(
; CHECK-NEXT: mul i8 %A, 6
; CHECK-NEXT: ret i8
        %a = mul i8 %A, 3
        %B = ashr exact i8 %a, 3
        %C = shl i8 %B, 4
        ret i8 %C
}

;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
define i32 @test14(i32 %A) {
; CHECK-LABEL: @test14(
; CHECK-NEXT: %B = and i32 %A, -19760
; CHECK-NEXT: or i32 %B, 19744
; CHECK-NEXT: ret i32
        %B = lshr i32 %A, 4             ; <i32> [#uses=1]
        %C = or i32 %B, 1234            ; <i32> [#uses=1]
        %D = shl i32 %C, 4              ; <i32> [#uses=1]
        ret i32 %D
}

;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
define i32 @test14a(i32 %A) {
; CHECK-LABEL: @test14a(
; CHECK-NEXT: and i32 %A, 77
; CHECK-NEXT: ret i32
        %B = shl i32 %A, 4              ; <i32> [#uses=1]
        %C = and i32 %B, 1234           ; <i32> [#uses=1]
        %D = lshr i32 %C, 4             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test15(i1 %C) {
; CHECK-LABEL: @test15(
; CHECK-NEXT: select i1 %C, i32 12, i32 4
; CHECK-NEXT: ret i32
        %A = select i1 %C, i32 3, i32 1         ; <i32> [#uses=1]
        %V = shl i32 %A, 2              ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test15a(i1 %C) {
; CHECK-LABEL: @test15a(
; CHECK-NEXT: select i1 %C, i32 512, i32 128
; CHECK-NEXT: ret i32
        %A = select i1 %C, i8 3, i8 1           ; <i8> [#uses=1]
        %shift.upgrd.4 = zext i8 %A to i32              ; <i32> [#uses=1]
        %V = shl i32 64, %shift.upgrd.4         ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test16(i32 %X) {
; CHECK-LABEL: @test16(
; CHECK-NEXT: and i32 %X, 16
; CHECK-NEXT: icmp ne i32
; CHECK-NEXT: ret i1
        %tmp.3 = ashr i32 %X, 4
        %tmp.6 = and i32 %tmp.3, 1
        %tmp.7 = icmp ne i32 %tmp.6, 0
        ret i1 %tmp.7
}

define i1 @test17(i32 %A) {
; CHECK-LABEL: @test17(
; CHECK-NEXT: and i32 %A, -8
; CHECK-NEXT: icmp eq i32
; CHECK-NEXT: ret i1
        %B = lshr i32 %A, 3             ; <i32> [#uses=1]
        %C = icmp eq i32 %B, 1234               ; <i1> [#uses=1]
        ret i1 %C
}


define i1 @test18(i8 %A) {
; CHECK-LABEL: @test18(
; CHECK: ret i1 false

        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        ;; false
        %C = icmp eq i8 %B, 123         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test19(i32 %A) {
; CHECK-LABEL: @test19(
; CHECK-NEXT: icmp ult i32 %A, 4
; CHECK-NEXT: ret i1
        %B = ashr i32 %A, 2             ; <i32> [#uses=1]
        ;; (X & -4) == 0
        %C = icmp eq i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}


define i1 @test19a(i32 %A) {
; CHECK-LABEL: @test19a(
; CHECK-NEXT: icmp ugt i32 %A, -5
; CHECK-NEXT: ret i1
        %B = ashr i32 %A, 2             ; <i32> [#uses=1]
        ;; X >u ~4
        %C = icmp eq i32 %B, -1         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test20(i8 %A) {
; CHECK-LABEL: @test20(
; CHECK: ret i1 false
        %B = ashr i8 %A, 7              ; <i8> [#uses=1]
        ;; false
        %C = icmp eq i8 %B, 123         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test21(i8 %A) {
; CHECK-LABEL: @test21(
; CHECK-NEXT: and i8 %A, 15
; CHECK-NEXT: icmp eq i8
; CHECK-NEXT: ret i1
        %B = shl i8 %A, 4               ; <i8> [#uses=1]
        %C = icmp eq i8 %B, -128                ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test22(i8 %A) {
; CHECK-LABEL: @test22(
; CHECK-NEXT: and i8 %A, 15
; CHECK-NEXT: icmp eq i8
; CHECK-NEXT: ret i1
        %B = shl i8 %A, 4               ; <i8> [#uses=1]
        %C = icmp eq i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %C
}

define i8 @test23(i32 %A) {
; CHECK-LABEL: @test23(
; CHECK-NEXT: trunc i32 %A to i8
; CHECK-NEXT: ret i8

        ;; casts not needed
        %B = shl i32 %A, 24             ; <i32> [#uses=1]
        %C = ashr i32 %B, 24            ; <i32> [#uses=1]
        %D = trunc i32 %C to i8         ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @test24(i8 %X) {
; CHECK-LABEL: @test24(
; CHECK-NEXT: and i8 %X, 3
; CHECK-NEXT: ret i8
        %Y = and i8 %X, -5              ; <i8> [#uses=1]
        %Z = shl i8 %Y, 5               ; <i8> [#uses=1]
        %Q = ashr i8 %Z, 5              ; <i8> [#uses=1]
        ret i8 %Q
}

define i32 @test25(i32 %tmp.2, i32 %AA) {
; CHECK-LABEL: @test25(
; CHECK-NEXT: and i32 %tmp.2, -131072
; CHECK-NEXT: add i32 %{{[^,]*}}, %AA
; CHECK-NEXT: and i32 %{{[^,]*}}, -131072
; CHECK-NEXT: ret i32
        %x = lshr i32 %AA, 17           ; <i32> [#uses=1]
        %tmp.3 = lshr i32 %tmp.2, 17            ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.3, %x             ; <i32> [#uses=1]
        %tmp.6 = shl i32 %tmp.5, 17             ; <i32> [#uses=1]
        ret i32 %tmp.6
}

define <2 x i32> @test25_vector(<2 x i32> %tmp.2, <2 x i32> %AA) {
; CHECK-LABEL: @test25_vector(
; CHECK: %tmp.3 = lshr <2 x i32> %tmp.2, <i32 17, i32 17>
; CHECK-NEXT: shl <2 x i32> %tmp.3, <i32 17, i32 17>
; CHECK-NEXT: add <2 x i32> %tmp.51, %AA
; CHECK-NEXT: and <2 x i32> %x2, <i32 -131072, i32 -131072>
; CHECK-NEXT: ret <2 x i32>
  %x = lshr <2 x i32> %AA, <i32 17, i32 17>
  %tmp.3 = lshr <2 x i32> %tmp.2, <i32 17, i32 17>
  %tmp.5 = add <2 x i32> %tmp.3, %x
  %tmp.6 = shl <2 x i32> %tmp.5, <i32 17, i32 17>
  ret <2 x i32> %tmp.6
}

;; handle casts between shifts.
define i32 @test26(i32 %A) {
; CHECK-LABEL: @test26(
; CHECK-NEXT: and i32 %A, -2
; CHECK-NEXT: ret i32
        %B = lshr i32 %A, 1             ; <i32> [#uses=1]
        %C = bitcast i32 %B to i32              ; <i32> [#uses=1]
        %D = shl i32 %C, 1              ; <i32> [#uses=1]
        ret i32 %D
}


define i1 @test27(i32 %x) nounwind {
; CHECK-LABEL: @test27(
; CHECK-NEXT: and i32 %x, 8
; CHECK-NEXT: icmp ne i32
; CHECK-NEXT: ret i1
  %y = lshr i32 %x, 3
  %z = trunc i32 %y to i1
  ret i1 %z
}

define i8 @test28(i8 %x) {
entry:
; CHECK-LABEL: @test28(
; CHECK:     icmp slt i8 %x, 0
; CHECK-NEXT:     br i1
	%tmp1 = lshr i8 %x, 7
	%cond1 = icmp ne i8 %tmp1, 0
	br i1 %cond1, label %bb1, label %bb2

bb1:
	ret i8 0

bb2:
	ret i8 1
}

define i8 @test28a(i8 %x, i8 %y) {
entry:
; This shouldn't be transformed.
; CHECK-LABEL: @test28a(
; CHECK:     %tmp1 = lshr i8 %x, 7
; CHECK:     %cond1 = icmp eq i8 %tmp1, 0
; CHECK:     br i1 %cond1, label %bb2, label %bb1
	%tmp1 = lshr i8 %x, 7
	%cond1 = icmp ne i8 %tmp1, 0
	br i1 %cond1, label %bb1, label %bb2
bb1:
	ret i8 %tmp1
bb2:
        %tmp2 = add i8 %tmp1, %y
	ret i8 %tmp2
}


define i32 @test29(i64 %d18) {
entry:
	%tmp916 = lshr i64 %d18, 32
	%tmp917 = trunc i64 %tmp916 to i32
	%tmp10 = lshr i32 %tmp917, 31
	ret i32 %tmp10
; CHECK-LABEL: @test29(
; CHECK:  %tmp916 = lshr i64 %d18, 63
; CHECK:  %tmp10 = trunc i64 %tmp916 to i32
}


define i32 @test30(i32 %A, i32 %B, i32 %C) {
	%X = shl i32 %A, %C
	%Y = shl i32 %B, %C
	%Z = and i32 %X, %Y
	ret i32 %Z
; CHECK-LABEL: @test30(
; CHECK: %X1 = and i32 %A, %B
; CHECK: %Z = shl i32 %X1, %C
}

define i32 @test31(i32 %A, i32 %B, i32 %C) {
	%X = lshr i32 %A, %C
	%Y = lshr i32 %B, %C
	%Z = or i32 %X, %Y
	ret i32 %Z
; CHECK-LABEL: @test31(
; CHECK: %X1 = or i32 %A, %B
; CHECK: %Z = lshr i32 %X1, %C
}

define i32 @test32(i32 %A, i32 %B, i32 %C) {
	%X = ashr i32 %A, %C
	%Y = ashr i32 %B, %C
	%Z = xor i32 %X, %Y
	ret i32 %Z
; CHECK-LABEL: @test32(
; CHECK: %X1 = xor i32 %A, %B
; CHECK: %Z = ashr i32 %X1, %C
; CHECK: ret i32 %Z
}

define i1 @test33(i32 %X) {
        %tmp1 = shl i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
; CHECK-LABEL: @test33(
; CHECK: %tmp1.mask = and i32 %X, 16777216
; CHECK: %tmp2 = icmp ne i32 %tmp1.mask, 0
}

define i1 @test34(i32 %X) {
        %tmp1 = lshr i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
; CHECK-LABEL: @test34(
; CHECK: ret i1 false
}

define i1 @test35(i32 %X) {
        %tmp1 = ashr i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
; CHECK-LABEL: @test35(
; CHECK: %tmp2 = icmp slt i32 %X, 0
; CHECK: ret i1 %tmp2
}

define i128 @test36(i128 %A, i128 %B) {
entry:
  %tmp27 = shl i128 %A, 64
  %tmp23 = shl i128 %B, 64
  %ins = or i128 %tmp23, %tmp27
  %tmp45 = lshr i128 %ins, 64
  ret i128 %tmp45

; CHECK-LABEL: @test36(
; CHECK:  %tmp231 = or i128 %B, %A
; CHECK:  %ins = and i128 %tmp231, 18446744073709551615
; CHECK:  ret i128 %ins
}

define i64 @test37(i128 %A, i32 %B) {
entry:
  %tmp27 = shl i128 %A, 64
  %tmp22 = zext i32 %B to i128
  %tmp23 = shl i128 %tmp22, 96
  %ins = or i128 %tmp23, %tmp27
  %tmp45 = lshr i128 %ins, 64
  %tmp46 = trunc i128 %tmp45 to i64
  ret i64 %tmp46

; CHECK-LABEL: @test37(
; CHECK:  %tmp23 = shl nuw nsw i128 %tmp22, 32
; CHECK:  %ins = or i128 %tmp23, %A
; CHECK:  %tmp46 = trunc i128 %ins to i64
}

define i32 @test38(i32 %x) nounwind readnone {
  %rem = srem i32 %x, 32
  %shl = shl i32 1, %rem
  ret i32 %shl
; CHECK-LABEL: @test38(
; CHECK-NEXT: and i32 %x, 31
; CHECK-NEXT: shl i32 1
; CHECK-NEXT: ret i32
}

; <rdar://problem/8756731>
; CHECK-LABEL: @test39(
define i8 @test39(i32 %a0) {
entry:
  %tmp4 = trunc i32 %a0 to i8
; CHECK: and i8 %tmp49, 64
  %tmp5 = shl i8 %tmp4, 5
  %tmp48 = and i8 %tmp5, 32
  %tmp49 = lshr i8 %tmp48, 5
  %tmp50 = mul i8 %tmp49, 64
  %tmp51 = xor i8 %tmp50, %tmp5
  %tmp52 = and i8 %tmp51, -128
  %tmp53 = lshr i8 %tmp52, 7
  %tmp54 = mul i8 %tmp53, 16
; CHECK: %0 = shl i8 %tmp4, 2
; CHECK: %tmp54 = and i8 %0, 16
  %tmp55 = xor i8 %tmp54, %tmp51
; CHECK: ret i8 %tmp55.1
  ret i8 %tmp55
}

; PR9809
define i32 @test40(i32 %a, i32 %b) nounwind {
  %shl1 = shl i32 1, %b
  %shl2 = shl i32 %shl1, 2
  %div = udiv i32 %a, %shl2
  ret i32 %div
; CHECK-LABEL: @test40(
; CHECK-NEXT: add i32 %b, 2
; CHECK-NEXT: lshr i32 %a
; CHECK-NEXT: ret i32
}

define i32 @test41(i32 %a, i32 %b) nounwind {
  %1 = shl i32 1, %b
  %2 = shl i32 %1, 3
  ret i32 %2
; CHECK-LABEL: @test41(
; CHECK-NEXT: shl i32 8, %b
; CHECK-NEXT: ret i32
}

define i32 @test42(i32 %a, i32 %b) nounwind {
  %div = lshr i32 4096, %b    ; must be exact otherwise we'd divide by zero
  %div2 = udiv i32 %a, %div
  ret i32 %div2
; CHECK-LABEL: @test42(
; CHECK-NEXT: lshr exact i32 4096, %b
}

define i32 @test43(i32 %a, i32 %b) nounwind {
  %div = shl i32 4096, %b    ; must be exact otherwise we'd divide by zero
  %div2 = udiv i32 %a, %div
  ret i32 %div2
; CHECK-LABEL: @test43(
; CHECK-NEXT: add i32 %b, 12
; CHECK-NEXT: lshr
; CHECK-NEXT: ret
}

define i32 @test44(i32 %a) nounwind {
  %y = shl nuw i32 %a, 1
  %z = shl i32 %y, 4
  ret i32 %z
; CHECK-LABEL: @test44(
; CHECK-NEXT: %y = shl i32 %a, 5
; CHECK-NEXT: ret i32 %y
}

define i32 @test45(i32 %a) nounwind {
  %y = lshr exact i32 %a, 1
  %z = lshr i32 %y, 4
  ret i32 %z
; CHECK-LABEL: @test45(
; CHECK-NEXT: %y = lshr i32 %a, 5
; CHECK-NEXT: ret i32 %y
}

define i32 @test46(i32 %a) {
  %y = ashr exact i32 %a, 3
  %z = shl i32 %y, 1
  ret i32 %z
; CHECK-LABEL: @test46(
; CHECK-NEXT: %z = ashr exact i32 %a, 2
; CHECK-NEXT: ret i32 %z
}

define i32 @test47(i32 %a) {
  %y = lshr exact i32 %a, 3
  %z = shl i32 %y, 1
  ret i32 %z
; CHECK-LABEL: @test47(
; CHECK-NEXT: %z = lshr exact i32 %a, 2
; CHECK-NEXT: ret i32 %z
}

define i32 @test48(i32 %x) {
  %A = lshr exact i32 %x, 1
  %B = shl i32 %A, 3
  ret i32 %B
; CHECK-LABEL: @test48(
; CHECK-NEXT: %B = shl i32 %x, 2
; CHECK-NEXT: ret i32 %B
}

define i32 @test49(i32 %x) {
  %A = ashr exact i32 %x, 1
  %B = shl i32 %A, 3
  ret i32 %B
; CHECK-LABEL: @test49(
; CHECK-NEXT: %B = shl i32 %x, 2
; CHECK-NEXT: ret i32 %B
}

define i32 @test50(i32 %x) {
  %A = shl nsw i32 %x, 1
  %B = ashr i32 %A, 3
  ret i32 %B
; CHECK-LABEL: @test50(
; CHECK-NEXT: %B = ashr i32 %x, 2
; CHECK-NEXT: ret i32 %B
}

define i32 @test51(i32 %x) {
  %A = shl nuw i32 %x, 1
  %B = lshr i32 %A, 3
  ret i32 %B
; CHECK-LABEL: @test51(
; CHECK-NEXT: %B = lshr i32 %x, 2
; CHECK-NEXT: ret i32 %B
}

define i32 @test52(i32 %x) {
  %A = shl nsw i32 %x, 3
  %B = ashr i32 %A, 1
  ret i32 %B
; CHECK-LABEL: @test52(
; CHECK-NEXT: %B = shl nsw i32 %x, 2
; CHECK-NEXT: ret i32 %B
}

define i32 @test53(i32 %x) {
  %A = shl nuw i32 %x, 3
  %B = lshr i32 %A, 1
  ret i32 %B
; CHECK-LABEL: @test53(
; CHECK-NEXT: %B = shl nuw i32 %x, 2
; CHECK-NEXT: ret i32 %B
}

define i32 @test54(i32 %x) {
  %shr2 = lshr i32 %x, 1
  %shl = shl i32 %shr2, 4
  %and = and i32 %shl, 16
  ret i32 %and
; CHECK-LABEL: @test54(
; CHECK: shl i32 %x, 3
}


define i32 @test55(i32 %x) {
  %shr2 = lshr i32 %x, 1
  %shl = shl i32 %shr2, 4
  %or = or i32 %shl, 8
  ret i32 %or
; CHECK-LABEL: @test55(
; CHECK: shl i32 %x, 3
}

define i32 @test56(i32 %x) {
  %shr2 = lshr i32 %x, 1
  %shl = shl i32 %shr2, 4
  %or = or i32 %shl, 7
  ret i32 %or
; CHECK-LABEL: @test56(
; CHECK: shl i32 %shr2, 4
}


define i32 @test57(i32 %x) {
  %shr = ashr i32 %x, 1
  %shl = shl i32 %shr, 4
  %or = or i32 %shl, 7
  ret i32 %or
; CHECK-LABEL: @test57(
; CHECK: %shl = shl i32 %shr.1, 4
}


define i32 @test58(i32 %x) {
  %shr = ashr i32 %x, 4
  %shl = shl i32 %shr, 1
  %or = or i32 %shl, 1
  ret i32 %or
; CHECK-LABEL: @test58(
; CHECK: ashr i32 %x, 3
}


define i32 @test59(i32 %x) {
  %shr = ashr i32 %x, 4
  %shl = shl i32 %shr, 1
  %or = or i32 %shl, 2
  ret i32 %or
; CHECK-LABEL: @test59(
; CHECK: ashr i32 %x, 4
}

; propagate "exact" trait
define i32 @test60(i32 %x) {
  %shr = ashr exact i32 %x, 4
  %shl = shl i32 %shr, 1
  %or = or i32 %shl, 1
  ret i32 %or
; CHECK-LABEL: @test60(
; CHECK: ashr exact i32 %x, 3
}

; PR17026
; CHECK-LABEL: @test61(
; CHECK-NOT: sh
; CHECK: ret
define void @test61(i128 %arg) {
bb:
  br i1 undef, label %bb1, label %bb12

bb1:                                              ; preds = %bb11, %bb
  br label %bb2

bb2:                                              ; preds = %bb7, %bb1
  br i1 undef, label %bb3, label %bb7

bb3:                                              ; preds = %bb2
  %tmp = lshr i128 %arg, 36893488147419103232
  %tmp4 = shl i128 %tmp, 0
  %tmp5 = or i128 %tmp4, undef
  %tmp6 = trunc i128 %tmp5 to i16
  br label %bb8

bb7:                                              ; preds = %bb2
  br i1 undef, label %bb8, label %bb2

bb8:                                              ; preds = %bb7, %bb3
  %tmp9 = phi i16 [ %tmp6, %bb3 ], [ undef, %bb7 ]
  %tmp10 = icmp eq i16 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb12

bb11:                                             ; preds = %bb8
  br i1 undef, label %bb1, label %bb12

bb12:                                             ; preds = %bb11, %bb8, %bb
  ret void
}

define i32 @test62(i32 %a) {
; CHECK-LABEL: @test62(
; CHECK-NEXT: ret i32 undef
  %b = ashr i32 %a, 32  ; shift all bits out
  ret i32 %b
}

define <4 x i32> @test62_splat_vector(<4 x i32> %a) {
; CHECK-LABEL: @test62_splat_vector
; CHECK-NEXT: ret <4 x i32> undef
  %b = ashr <4 x i32> %a, <i32 32, i32 32, i32 32, i32 32>  ; shift all bits out
  ret <4 x i32> %b
}

define <4 x i32> @test62_non_splat_vector(<4 x i32> %a) {
; CHECK-LABEL: @test62_non_splat_vector
; CHECK-NOT: ret <4 x i32> undef
  %b = ashr <4 x i32> %a, <i32 32, i32 0, i32 1, i32 2>  ; shift all bits out
  ret <4 x i32> %b
}

define <2 x i65> @test_63(<2 x i64> %t) {
; CHECK-LABEL: @test_63
  %a = zext <2 x i64> %t to <2 x i65>
  %sext = shl <2 x i65> %a, <i65 33, i65 33>
  %b = ashr <2 x i65> %sext, <i65 33, i65 33>
  ret <2 x i65> %b
}
