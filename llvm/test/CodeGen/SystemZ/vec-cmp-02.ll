; Test v8i16 comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test eq.
define <8 x i16> @f1(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f1:
; CHECK: vceqh %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp eq <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test ne.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vceqh [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test sgt.
define <8 x i16> @f3(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f3:
; CHECK: vchh %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test sge.
define <8 x i16> @f4(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f4:
; CHECK: vchh [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test sle.
define <8 x i16> @f5(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f5:
; CHECK: vchh [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test slt.
define <8 x i16> @f6(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f6:
; CHECK: vchh %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp slt <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test ugt.
define <8 x i16> @f7(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f7:
; CHECK: vchlh %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test uge.
define <8 x i16> @f8(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f8:
; CHECK: vchlh [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test ule.
define <8 x i16> @f9(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f9:
; CHECK: vchlh [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test ult.
define <8 x i16> @f10(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f10:
; CHECK: vchlh %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = icmp ult <8 x i16> %val1, %val2
  %ret = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %ret
}

; Test eq selects.
define <8 x i16> @f11(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f11:
; CHECK: vceqh [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp eq <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test ne selects.
define <8 x i16> @f12(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f12:
; CHECK: vceqh [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ne <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test sgt selects.
define <8 x i16> @f13(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f13:
; CHECK: vchh [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sgt <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test sge selects.
define <8 x i16> @f14(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f14:
; CHECK: vchh [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sge <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test sle selects.
define <8 x i16> @f15(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f15:
; CHECK: vchh [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp sle <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test slt selects.
define <8 x i16> @f16(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f16:
; CHECK: vchh [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp slt <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test ugt selects.
define <8 x i16> @f17(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f17:
; CHECK: vchlh [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ugt <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test uge selects.
define <8 x i16> @f18(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f18:
; CHECK: vchlh [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp uge <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test ule selects.
define <8 x i16> @f19(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f19:
; CHECK: vchlh [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ule <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}

; Test ult selects.
define <8 x i16> @f20(<8 x i16> %val1, <8 x i16> %val2,
                      <8 x i16> %val3, <8 x i16> %val4) {
; CHECK-LABEL: f20:
; CHECK: vchlh [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = icmp ult <8 x i16> %val1, %val2
  %ret = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %ret
}
