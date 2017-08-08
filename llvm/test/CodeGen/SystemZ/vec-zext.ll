; Test that vector zexts are done efficently with unpack instructions also in
; case of fewer elements than allowed, e.g. <2 x i32>.
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
; CHECK:      	vuplhb	%v0, %v24
; CHECK-NEXT: 	vuplhh	%v24, %v0
; CHECK-NEXT: 	br	%r14
  %z = zext <2 x i8> %val1 to <2 x i32>
  ret <2 x i32> %z
}

define <2 x i64> @fun3(<2 x i8> %val1) {
; CHECK-LABEL: fun3:
; CHECK:      	vuplhb	%v0, %v24
; CHECK-NEXT: 	vuplhh	%v0, %v0
; CHECK-NEXT: 	vuplhf	%v24, %v0
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
; CHECK:      	vuplhh	%v0, %v24
; CHECK-NEXT: 	vuplhf	%v24, %v0
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
; CHECK:      	vuplhb	%v0, %v24
; CHECK-NEXT: 	vuplhh	%v24, %v0
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

