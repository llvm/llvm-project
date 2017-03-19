; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @identity_bitreverse_i32(i32 %p) {
; CHECK-LABEL: @identity_bitreverse_i32(
; CHECK-NEXT: ret i32 %p
  %a = call i32 @llvm.bitreverse.i32(i32 %p)
  %b = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %b
}

; CHECK-LABEL: @identity_bitreverse_v2i32(
; CHECK-NEXT: ret <2 x i32> %p
define <2 x i32> @identity_bitreverse_v2i32(<2 x i32> %p) {
  %a = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %p)
  %b = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %a)
  ret <2 x i32> %b
}

; CHECK-LABEL: @reverse_0_i32(
; CHECK-NEXT: ret i32 0
define i32 @reverse_0_i32() {
  %x = call i32 @llvm.bitreverse.i32(i32 0)
  ret i32 %x
}

; CHECK-LABEL: @reverse_1_i32(
; CHECK-NEXT: ret i32 -2147483648
define i32 @reverse_1_i32() {
  %x = call i32 @llvm.bitreverse.i32(i32 1)
  ret i32 %x
}

; CHECK-LABEL: @reverse_neg1_i32(
; CHECK-NEXT: ret i32 -1
define i32 @reverse_neg1_i32() {
  %x = call i32 @llvm.bitreverse.i32(i32 -1)
  ret i32 %x
}

; CHECK-LABEL: @reverse_undef_i32(
; CHECK-NEXT: ret i32 undef
define i32 @reverse_undef_i32() {
  %x = call i32 @llvm.bitreverse.i32(i32 undef)
  ret i32 %x
}

; CHECK-LABEL: @reverse_false_i1(
; CHECK-NEXT: ret i1 false
define i1 @reverse_false_i1() {
  %x = call i1 @llvm.bitreverse.i1(i1 false)
  ret i1 %x
}

; CHECK-LABEL: @reverse_true_i1(
; CHECK-NEXT: ret i1 true
define i1 @reverse_true_i1() {
  %x = call i1 @llvm.bitreverse.i1(i1 true)
  ret i1 %x
}

; CHECK-LABEL: @reverse_undef_i1(
; CHECK-NEXT: ret i1 undef
define i1 @reverse_undef_i1() {
  %x = call i1 @llvm.bitreverse.i1(i1 undef)
  ret i1 %x
}

; CHECK-LABEL: @reverse_false_v2i1(
; CHECK-NEXT: ret <2 x i1> zeroinitializer
define <2 x i1> @reverse_false_v2i1() {
  %x = call <2  x i1> @llvm.bitreverse.v2i1(<2 x i1> zeroinitializer)
  ret <2 x i1> %x
}

; CHECK-LABEL: @reverse_true_v2i1(
; CHECK-NEXT: ret <2 x i1> <i1 true, i1 true>
define <2 x i1> @reverse_true_v2i1() {
  %x = call <2 x i1> @llvm.bitreverse.v2i1(<2 x i1> <i1 true, i1 true>)
  ret <2 x i1> %x
}

; CHECK-LABEL: @bitreverse_920_1234_v2i32(
; CHECK-NEXT: ret <2 x i32> <i32 432013312, i32 1260388352>
define <2 x i32> @bitreverse_920_1234_v2i32() {
  %x = call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> <i32 920, i32 1234>)
  ret <2 x i32> %x
}

; CHECK-LABEL: @reverse_100_i3(
; CHECK-NEXT: ret i3 1
define i3 @reverse_100_i3() {
  %x = call i3 @llvm.bitreverse.i3(i3 100)
  ret i3 %x
}

; CHECK-LABEL: @reverse_6_3_v2i3(
; CHECK-NEXT: ret <2 x i3> <i3 3, i3 -2>
define <2 x i3> @reverse_6_3_v2i3() {
  %x = call <2  x i3> @llvm.bitreverse.v2i3(<2 x i3> <i3 6, i3 3>)
  ret <2 x i3> %x
}

declare i1 @llvm.bitreverse.i1(i1) readnone
declare <2 x i1> @llvm.bitreverse.v2i1(<2 x i1>) readnone

declare i3 @llvm.bitreverse.i3(i3) readnone
declare <2 x i3> @llvm.bitreverse.v2i3(<2 x i3>) readnone

declare i32 @llvm.bitreverse.i32(i32) readnone
declare <2 x i32> @llvm.bitreverse.v2i32(<2 x i32>) readnone
