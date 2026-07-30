; Test v2i64 comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test eq.
define <2 x i64> @f1(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f1:
; CHECK: vceqg %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp eq <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ne.
define <2 x i64> @f2(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f2:
; CHECK: vceqg [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test sgt.
define <2 x i64> @f3(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f3:
; CHECK: vchg %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test sge.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vchg [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test sle.
define <2 x i64> @f5(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f5:
; CHECK: vchg [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test slt.
define <2 x i64> @f6(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f6:
; CHECK: vchg %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp slt <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ugt.
define <2 x i64> @f7(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f7:
; CHECK: vchlg %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test uge.
define <2 x i64> @f8(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f8:
; CHECK: vchlg [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ule.
define <2 x i64> @f9(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f9:
; CHECK: vchlg [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ult.
define <2 x i64> @f10(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f10:
; CHECK: vchlg %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp ult <2 x i64> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test eq selects.
define <2 x i64> @f11(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f11:
; CHECK: vceqg [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp eq <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test ne selects.
define <2 x i64> @f12(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f12:
; CHECK: vceqg [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test sgt selects.
define <2 x i64> @f13(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f13:
; CHECK: vchg [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test sge selects.
define <2 x i64> @f14(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f14:
; CHECK: vchg [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test sle selects.
define <2 x i64> @f15(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f15:
; CHECK: vchg [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test slt selects.
define <2 x i64> @f16(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f16:
; CHECK: vchg [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp slt <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test ugt selects.
define <2 x i64> @f17(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f17:
; CHECK: vchlg [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test uge selects.
define <2 x i64> @f18(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f18:
; CHECK: vchlg [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test ule selects.
define <2 x i64> @f19(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f19:
; CHECK: vchlg [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}

; Test ult selects.
define <2 x i64> @f20(<2 x i64> %val1, <2 x i64> %val2,
                      <2 x i64> %val3, <2 x i64> %val4) {
; CHECK-LABEL: f20:
; CHECK: vchlg [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ult <2 x i64> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %ret
}
