; Test v4i32 comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test eq.
define <4 x i32> @f1(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f1:
; CHECK: vceqf %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp eq <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ne.
define <4 x i32> @f2(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f2:
; CHECK: vceqf [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test sgt.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vchf %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test sge.
define <4 x i32> @f4(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f4:
; CHECK: vchf [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test sle.
define <4 x i32> @f5(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f5:
; CHECK: vchf [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test slt.
define <4 x i32> @f6(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f6:
; CHECK: vchf %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp slt <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ugt.
define <4 x i32> @f7(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f7:
; CHECK: vchlf %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test uge.
define <4 x i32> @f8(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f8:
; CHECK: vchlf [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ule.
define <4 x i32> @f9(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f9:
; CHECK: vchlf [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ult.
define <4 x i32> @f10(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f10:
; CHECK: vchlf %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp ult <4 x i32> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test eq selects.
define <4 x i32> @f11(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f11:
; CHECK: vceqf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp eq <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test ne selects.
define <4 x i32> @f12(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f12:
; CHECK: vceqf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test sgt selects.
define <4 x i32> @f13(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f13:
; CHECK: vchf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test sge selects.
define <4 x i32> @f14(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f14:
; CHECK: vchf [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test sle selects.
define <4 x i32> @f15(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f15:
; CHECK: vchf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test slt selects.
define <4 x i32> @f16(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f16:
; CHECK: vchf [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp slt <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test ugt selects.
define <4 x i32> @f17(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f17:
; CHECK: vchlf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test uge selects.
define <4 x i32> @f18(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f18:
; CHECK: vchlf [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test ule selects.
define <4 x i32> @f19(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f19:
; CHECK: vchlf [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}

; Test ult selects.
define <4 x i32> @f20(<4 x i32> %val1, <4 x i32> %val2,
                      <4 x i32> %val3, <4 x i32> %val4) {
; CHECK-LABEL: f20:
; CHECK: vchlf [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ult <4 x i32> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %ret
}
