; Test v8i16 minimum.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <8 x i16> @f1(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmnh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp slt <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val1
  ret <8 x i16> %ret
}

; Test with sle.
define <8 x i16> @f2(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vmnh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sle <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val1
  ret <8 x i16> %ret
}

; Test with sgt.
define <8 x i16> @f3(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f3:
; CHECK: vmnh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sgt <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val1, <8 x i16> %val2
  ret <8 x i16> %ret
}

; Test with sge.
define <8 x i16> @f4(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f4:
; CHECK: vmnh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sge <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val1, <8 x i16> %val2
  ret <8 x i16> %ret
}

; Test with ult.
define <8 x i16> @f5(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f5:
; CHECK: vmnlh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ult <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val1
  ret <8 x i16> %ret
}

; Test with ule.
define <8 x i16> @f6(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f6:
; CHECK: vmnlh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ule <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val2, <8 x i16> %val1
  ret <8 x i16> %ret
}

; Test with ugt.
define <8 x i16> @f7(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f7:
; CHECK: vmnlh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ugt <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val1, <8 x i16> %val2
  ret <8 x i16> %ret
}

; Test with uge.
define <8 x i16> @f8(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f8:
; CHECK: vmnlh %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp uge <8 x i16> %val2, %val1
  %ret = select <8 x i1> %cmp, <8 x i16> %val1, <8 x i16> %val2
  ret <8 x i16> %ret
}
