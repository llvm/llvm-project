; Test that vector zexts are done efficently also in case of fewer elements
; than allowed, e.g. <2 x i32>.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s


define <2 x i16> @fun1(<2 x i8> %val1) {
; CHECK-LABEL: fun1:
; CHECK:      	vuplhb	%v24, %v24
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i8> %val1 to <2 x i16>
  ret <2 x i16> %z
}

define <2 x i32> @fun2(<2 x i8> %val1) {
; CHECK-LABEL: fun2:
; CHECK:        larl	%r1, .LCPI1_0
; CHECK-NEXT:   vl	%v0, 0(%r1), 3
; CHECK-NEXT:   vperm	%v24, %v0, %v24, %v0
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i8> %val1 to <2 x i32>
  ret <2 x i32> %z
}

define <2 x i64> @fun3(<2 x i8> %val1) {
; CHECK-LABEL: fun3:
; CHECK: 	larl	%r1, .LCPI2_0
; CHECK-NEXT: 	vl	%v0, 0(%r1), 3
; CHECK-NEXT: 	vperm	%v24, %v0, %v24, %v0
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i8> %val1 to <2 x i64>
  ret <2 x i64> %z
}

define <2 x i32> @fun4(<2 x i16> %val1) {
; CHECK-LABEL: fun4:
; CHECK:      	vuplhh	%v24, %v24
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i16> %val1 to <2 x i32>
  ret <2 x i32> %z
}

define <2 x i64> @fun5(<2 x i16> %val1) {
; CHECK-LABEL: fun5:
; CHECK: 	larl	%r1, .LCPI4_0
; CHECK-NEXT: 	vl	%v0, 0(%r1), 3
; CHECK-NEXT: 	vperm	%v24, %v0, %v24, %v0
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i16> %val1 to <2 x i64>
  ret <2 x i64> %z
}

define <2 x i64> @fun6(<2 x i32> %val1) {
; CHECK-LABEL: fun6:
; CHECK:      	vuplhf	%v24, %v24
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i32> %val1 to <2 x i64>
  ret <2 x i64> %z
}

define <4 x i16> @fun7(<4 x i8> %val1) {
; CHECK-LABEL: fun7:
; CHECK:      	vuplhb	%v24, %v24
; CHECK-NEXT: 	br	%r14
  %z = zext <4 x i8> %val1 to <4 x i16>
  ret <4 x i16> %z
}

define <4 x i32> @fun8(<4 x i8> %val1) {
; CHECK-LABEL: fun8:
; CHECK: 	larl	%r1, .LCPI7_0
; CHECK-NEXT: 	vl	%v0, 0(%r1), 3
; CHECK-NEXT: 	vperm	%v24, %v0, %v24, %v0
; CHECK-NEXT: 	br	%r14
  %z = zext <4 x i8> %val1 to <4 x i32>
  ret <4 x i32> %z
}

define <4 x i32> @fun9(<4 x i16> %val1) {
; CHECK-LABEL: fun9:
; CHECK:      	vuplhh	%v24, %v24
; CHECK-NEXT: 	br	%r14
  %z = zext <4 x i16> %val1 to <4 x i32>
  ret <4 x i32> %z
}

define <8 x i16> @fun10(<8 x i8> %val1) {
; CHECK-LABEL: fun10:
; CHECK:      	vuplhb	%v24, %v24
; CHECK-NEXT: 	br	%r14
  %z = zext <8 x i8> %val1 to <8 x i16>
  ret <8 x i16> %z
}

define <2 x i32> @fun11(<2 x i64> %Arg1, <2 x i64> %Arg2) {
; CHECK-LABEL: fun11:
; CHECK:      vgbm    %v0, 0
; CHECK-NEXT: vceqg   %v1, %v24, %v0
; CHECK-NEXT: vceqg   %v0, %v26, %v0
; CHECK-NEXT: vo      %v0, %v1, %v0
; CHECK-NEXT: vrepig  %v1, 1
; CHECK-NEXT: vn      %v0, %v0, %v1
; CHECK-NEXT: vpkg    %v24, %v0, %v0
; CHECK-NEXT: br      %r14
  %i3 = icmp eq <2 x i64> %Arg1, zeroinitializer
  %i5 = icmp eq <2 x i64> %Arg2, zeroinitializer
  %i6 = or <2 x i1> %i3, %i5
  %i7 = zext <2 x i1> %i6 to <2 x i32>
  ret <2 x i32> %i7
}
