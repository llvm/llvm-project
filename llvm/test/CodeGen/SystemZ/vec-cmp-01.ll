; Test v16i8 comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test eq.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vceqb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp eq <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test ne.
define <16 x i8> @f2(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f2:
; CHECK: vceqb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test sgt.
define <16 x i8> @f3(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f3:
; CHECK: vchb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test sge.
define <16 x i8> @f4(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f4:
; CHECK: vchb [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test sle.
define <16 x i8> @f5(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f5:
; CHECK: vchb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test slt.
define <16 x i8> @f6(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f6:
; CHECK: vchb %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp slt <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test ugt.
define <16 x i8> @f7(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f7:
; CHECK: vchlb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test uge.
define <16 x i8> @f8(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f8:
; CHECK: vchlb [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test ule.
define <16 x i8> @f9(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f9:
; CHECK: vchlb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test ult.
define <16 x i8> @f10(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f10:
; CHECK: vchlb %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp ult <16 x i8> %val1, %val2
  %ret = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %ret
}

; Test eq selects.
define <16 x i8> @f11(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f11:
; CHECK: vceqb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp eq <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test ne selects.
define <16 x i8> @f12(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f12:
; CHECK: vceqb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test sgt selects.
define <16 x i8> @f13(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f13:
; CHECK: vchb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test sge selects.
define <16 x i8> @f14(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f14:
; CHECK: vchb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test sle selects.
define <16 x i8> @f15(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f15:
; CHECK: vchb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test slt selects.
define <16 x i8> @f16(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f16:
; CHECK: vchb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp slt <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test ugt selects.
define <16 x i8> @f17(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f17:
; CHECK: vchlb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test uge selects.
define <16 x i8> @f18(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f18:
; CHECK: vchlb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test ule selects.
define <16 x i8> @f19(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f19:
; CHECK: vchlb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}

; Test ult selects.
define <16 x i8> @f20(<16 x i8> %val1, <16 x i8> %val2,
                      <16 x i8> %val3, <16 x i8> %val4) {
; CHECK-LABEL: f20:
; CHECK: vchlb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ult <16 x i8> %val1, %val2
  %ret = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %ret
}
