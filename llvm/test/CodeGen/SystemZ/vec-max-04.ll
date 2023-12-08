; Test v2i64 maximum.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test with slt.
define <2 x i64> @f1(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmxg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp slt <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val1
  ret <2 x i64> %ret
}

; Test with sle.
define <2 x i64> @f2(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f2:
; CHECK: vmxg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sle <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val1
  ret <2 x i64> %ret
}

; Test with sgt.
define <2 x i64> @f3(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f3:
; CHECK: vmxg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sgt <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val1, <2 x i64> %val2
  ret <2 x i64> %ret
}

; Test with sge.
define <2 x i64> @f4(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vmxg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp sge <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val1, <2 x i64> %val2
  ret <2 x i64> %ret
}

; Test with ult.
define <2 x i64> @f5(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f5:
; CHECK: vmxlg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ult <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val1
  ret <2 x i64> %ret
}

; Test with ule.
define <2 x i64> @f6(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f6:
; CHECK: vmxlg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ule <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val2, <2 x i64> %val1
  ret <2 x i64> %ret
}

; Test with ugt.
define <2 x i64> @f7(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f7:
; CHECK: vmxlg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp ugt <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val1, <2 x i64> %val2
  ret <2 x i64> %ret
}

; Test with uge.
define <2 x i64> @f8(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f8:
; CHECK: vmxlg %v24, {{%v24, %v26|%v26, %v24}}
; CHECK: br %r14
  %cmp = icmp uge <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val1, <2 x i64> %val2
  ret <2 x i64> %ret
}
