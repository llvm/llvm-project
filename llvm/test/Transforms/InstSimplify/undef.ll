; RUN: opt -instsimplify -S < %s | FileCheck %s

; @test0
; CHECK: ret i64 undef
define i64 @test0() {
  %r = mul i64 undef, undef
  ret i64 %r
}

; @test1
; CHECK: ret i64 undef
define i64 @test1() {
  %r = mul i64 3, undef
  ret i64 %r
}

; @test2
; CHECK: ret i64 undef
define i64 @test2() {
  %r = mul i64 undef, 3
  ret i64 %r
}

; @test3
; CHECK: ret i64 0
define i64 @test3() {
  %r = mul i64 undef, 6
  ret i64 %r
}

; @test4
; CHECK: ret i64 0
define i64 @test4() {
  %r = mul i64 6, undef
  ret i64 %r
}

; @test5
; CHECK: ret i64 undef
define i64 @test5() {
  %r = and i64 undef, undef
  ret i64 %r
}

; @test6
; CHECK: ret i64 undef
define i64 @test6() {
  %r = or i64 undef, undef
  ret i64 %r
}

; @test7
; CHECK: ret i64 undef
define i64 @test7() {
  %r = udiv i64 undef, 1
  ret i64 %r
}

; @test8
; CHECK: ret i64 undef
define i64 @test8() {
  %r = sdiv i64 undef, 1
  ret i64 %r
}

; @test9
; CHECK: ret i64 0
define i64 @test9() {
  %r = urem i64 undef, 1
  ret i64 %r
}

; @test10
; CHECK: ret i64 0
define i64 @test10() {
  %r = srem i64 undef, 1
  ret i64 %r
}

; @test11
; CHECK: ret i64 undef
define i64 @test11() {
  %r = shl i64 undef, undef
  ret i64 %r
}

; @test11b
; CHECK: ret i64 undef
define i64 @test11b(i64 %a) {
  %r = shl i64 %a, undef
  ret i64 %r
}

; @test12
; CHECK: ret i64 undef
define i64 @test12() {
  %r = ashr i64 undef, undef
  ret i64 %r
}

; @test12b
; CHECK: ret i64 undef
define i64 @test12b(i64 %a) {
  %r = ashr i64 %a, undef
  ret i64 %r
}

; @test13
; CHECK: ret i64 undef
define i64 @test13() {
  %r = lshr i64 undef, undef
  ret i64 %r
}

; @test13b
; CHECK: ret i64 undef
define i64 @test13b(i64 %a) {
  %r = lshr i64 %a, undef
  ret i64 %r
}

; @test14
; CHECK: ret i1 undef
define i1 @test14() {
  %r = icmp slt i64 undef, undef
  ret i1 %r
}

; @test15
; CHECK: ret i1 undef
define i1 @test15() {
  %r = icmp ult i64 undef, undef
  ret i1 %r
}

; @test16
; CHECK: ret i64 undef
define i64 @test16(i64 %a) {
  %r = select i1 undef, i64 %a, i64 undef
  ret i64 %r
}

; @test17
; CHECK: ret i64 undef
define i64 @test17(i64 %a) {
  %r = select i1 undef, i64 undef, i64 %a
  ret i64 %r
}

; @test18
; CHECK: ret i64 undef
define i64 @test18(i64 %a) {
  %r = call i64 (i64) undef(i64 %a)
  ret i64 %r
}

; CHECK-LABEL: @test19
; CHECK: ret <4 x i8> undef
define <4 x i8> @test19(<4 x i8> %a) {
  %b = shl <4 x i8> %a, <i8 8, i8 9, i8 undef, i8 -1>
  ret <4 x i8> %b
}

; CHECK-LABEL: @test20
; CHECK: ret i32 undef
define i32 @test20(i32 %a) {
  %b = udiv i32 %a, 0
  ret i32 %b
}

; CHECK-LABEL: @test21
; CHECK: ret i32 undef
define i32 @test21(i32 %a) {
  %b = sdiv i32 %a, 0
  ret i32 %b
}

; CHECK-LABEL: @test22
; CHECK: ret i32 undef
define i32 @test22(i32 %a) {
  %b = ashr exact i32 undef, %a
  ret i32 %b
}

; CHECK-LABEL: @test23
; CHECK: ret i32 undef
define i32 @test23(i32 %a) {
  %b = lshr exact i32 undef, %a
  ret i32 %b
}

; CHECK-LABEL: @test24
; CHECK: ret i32 undef
define i32 @test24() {
  %b = udiv i32 undef, 0
  ret i32 %b
}

; CHECK-LABEL: @test25
; CHECK: ret i32 undef
define i32 @test25() {
  %b = lshr i32 0, undef
  ret i32 %b
}

; CHECK-LABEL: @test26
; CHECK: ret i32 undef
define i32 @test26() {
  %b = ashr i32 0, undef
  ret i32 %b
}

; CHECK-LABEL: @test27
; CHECK: ret i32 undef
define i32 @test27() {
  %b = shl i32 0, undef
  ret i32 %b
}

; CHECK-LABEL: @test28
; CHECK: ret i32 undef
define i32 @test28(i32 %a) {
  %b = shl nsw i32 undef, %a
  ret i32 %b
}

; CHECK-LABEL: @test29
; CHECK: ret i32 undef
define i32 @test29(i32 %a) {
  %b = shl nuw i32 undef, %a
  ret i32 %b
}

; CHECK-LABEL: @test30
; CHECK: ret i32 undef
define i32 @test30(i32 %a) {
  %b = shl nsw nuw i32 undef, %a
  ret i32 %b
}

; CHECK-LABEL: @test31
; CHECK: ret i32 0
define i32 @test31(i32 %a) {
  %b = shl i32 undef, %a
  ret i32 %b
}

; CHECK-LABEL: @test32
; CHECK: ret i32 undef
define i32 @test32(i32 %a) {
  %b = shl i32 undef, 0
  ret i32 %b
}

; CHECK-LABEL: @test33
; CHECK: ret i32 undef
define i32 @test33(i32 %a) {
  %b = ashr i32 undef, 0
  ret i32 %b
}

; CHECK-LABEL: @test34
; CHECK: ret i32 undef
define i32 @test34(i32 %a) {
  %b = lshr i32 undef, 0
  ret i32 %b
}

; CHECK-LABEL: @test35
; CHECK: ret i32 undef
define i32 @test35(<4 x i32> %V) {
  %b = extractelement <4 x i32> %V, i32 4
  ret i32 %b
}

; CHECK-LABEL: @test36
; CHECK: ret i32 undef
define i32 @test36(i32 %V) {
  %b = extractelement <4 x i32> undef, i32 %V
  ret i32 %b
}

; CHECK-LABEL: @test37
; CHECK: ret i32 undef
define i32 @test37() {
  %b = udiv i32 undef, undef
  ret i32 %b
}

; CHECK-LABEL: @test38
; CHECK: ret i32 undef
define i32 @test38(i32 %a) {
  %b = udiv i32 %a, undef
  ret i32 %b
}

; CHECK-LABEL: @test39
; CHECK: ret i32 undef
define i32 @test39() {
  %b = udiv i32 0, undef
  ret i32 %b
}
