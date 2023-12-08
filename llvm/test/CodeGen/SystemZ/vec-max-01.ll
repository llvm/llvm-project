; Test v16i8 maximum.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmxb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp slt <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val1
  ret <16 x i8> %ret
}

; Test with sle.
define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f2:
; CHECK: vmxb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sle <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val1
  ret <16 x i8> %ret
}

; Test with sgt.
define <16 x i8> @f3(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f3:
; CHECK: vmxb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sgt <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val1, <16 x i8> %val2
  ret <16 x i8> %ret
}

; Test with sge.
define <16 x i8> @f4(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f4:
; CHECK: vmxb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sge <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val1, <16 x i8> %val2
  ret <16 x i8> %ret
}

; Test with ult.
define <16 x i8> @f5(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f5:
; CHECK: vmxlb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ult <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val1
  ret <16 x i8> %ret
}

; Test with ule.
define <16 x i8> @f6(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f6:
; CHECK: vmxlb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ule <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val2, <16 x i8> %val1
  ret <16 x i8> %ret
}

; Test with ugt.
define <16 x i8> @f7(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f7:
; CHECK: vmxlb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ugt <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val1, <16 x i8> %val2
  ret <16 x i8> %ret
}

; Test with uge.
define <16 x i8> @f8(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f8:
; CHECK: vmxlb %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp uge <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val1, <16 x i8> %val2
  ret <16 x i8> %ret
}
