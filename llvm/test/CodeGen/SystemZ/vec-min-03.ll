; Test v4i32 minimum.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <4 x i32> @f1(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmnf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp slt <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val1
  ret <4 x i32> %ret
}

; Test with sle.
define <4 x i32> @f2(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f2:
; CHECK: vmnf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sle <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val1
  ret <4 x i32> %ret
}

; Test with sgt.
define <4 x i32> @f3(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vmnf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sgt <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val1, <4 x i32> %val2
  ret <4 x i32> %ret
}

; Test with sge.
define <4 x i32> @f4(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f4:
; CHECK: vmnf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sge <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val1, <4 x i32> %val2
  ret <4 x i32> %ret
}

; Test with ult.
define <4 x i32> @f5(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f5:
; CHECK: vmnlf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ult <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val1
  ret <4 x i32> %ret
}

; Test with ule.
define <4 x i32> @f6(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f6:
; CHECK: vmnlf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ule <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val2, <4 x i32> %val1
  ret <4 x i32> %ret
}

; Test with ugt.
define <4 x i32> @f7(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f7:
; CHECK: vmnlf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ugt <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val1, <4 x i32> %val2
  ret <4 x i32> %ret
}

; Test with uge.
define <4 x i32> @f8(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f8:
; CHECK: vmnlf %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp uge <4 x i32> %val2, %val1
  %ret = select <4 x i1> %cmp, <4 x i32> %val1, <4 x i32> %val2
  ret <4 x i32> %ret
}
