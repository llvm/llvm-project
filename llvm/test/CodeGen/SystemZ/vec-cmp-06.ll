; Test f64 and v2f64 comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test oeq.
define <2 x i64> @f1(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f1:
; CHECK: vfcedb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = fcmp oeq <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test one.
define <2 x i64> @f2(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f2:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfchdb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vo %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = fcmp one <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ogt.
define <2 x i64> @f3(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f3:
; CHECK: vfchdb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test oge.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f4:
; CHECK: vfchedb %v24, %v26, %v28
; CHECK-NEXT: br %r14
  %cmp = fcmp oge <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ole.
define <2 x i64> @f5(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f5:
; CHECK: vfchedb %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = fcmp ole <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test olt.
define <2 x i64> @f6(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: vfchdb %v24, %v28, %v26
; CHECK-NEXT: br %r14
  %cmp = fcmp olt <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ueq.
define <2 x i64> @f7(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f7:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfchdb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vno %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ueq <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test une.
define <2 x i64> @f8(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f8:
; CHECK: vfcedb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp une <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ugt.
define <2 x i64> @f9(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f9:
; CHECK: vfchedb [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ugt <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test uge.
define <2 x i64> @f10(<2 x i64> %dummy, <2 x double> %val1,
                      <2 x double> %val2) {
; CHECK-LABEL: f10:
; CHECK: vfchdb [[REG:%v[0-9]+]], %v28, %v26
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uge <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ule.
define <2 x i64> @f11(<2 x i64> %dummy, <2 x double> %val1,
                      <2 x double> %val2) {
; CHECK-LABEL: f11:
; CHECK: vfchdb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ule <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ult.
define <2 x i64> @f12(<2 x i64> %dummy, <2 x double> %val1,
                      <2 x double> %val2) {
; CHECK-LABEL: f12:
; CHECK: vfchedb [[REG:%v[0-9]+]], %v26, %v28
; CHECK-NEXT: vno %v24, [[REG]], [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ult <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test ord.
define <2 x i64> @f13(<2 x i64> %dummy, <2 x double> %val1,
                      <2 x double> %val2) {
; CHECK-LABEL: f13:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfchedb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vo %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ord <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test uno.
define <2 x i64> @f14(<2 x i64> %dummy, <2 x double> %val1,
                      <2 x double> %val2) {
; CHECK-LABEL: f14:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v28, %v26
; CHECK-DAG: vfchedb [[REG2:%v[0-9]+]], %v26, %v28
; CHECK: vno %v24, [[REG1]], [[REG2]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uno <2 x double> %val1, %val2
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test oeq selects.
define <2 x double> @f15(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f15:
; CHECK: vfcedb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp oeq <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test one selects.
define <2 x double> @f16(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f16:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfchdb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp one <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ogt selects.
define <2 x double> @f17(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f17:
; CHECK: vfchdb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test oge selects.
define <2 x double> @f18(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f18:
; CHECK: vfchedb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp oge <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ole selects.
define <2 x double> @f19(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f19:
; CHECK: vfchedb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ole <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test olt selects.
define <2 x double> @f20(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f20:
; CHECK: vfchdb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp olt <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ueq selects.
define <2 x double> @f21(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f21:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfchdb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ueq <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test une selects.
define <2 x double> @f22(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f22:
; CHECK: vfcedb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp une <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ugt selects.
define <2 x double> @f23(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f23:
; CHECK: vfchedb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ugt <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test uge selects.
define <2 x double> @f24(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f24:
; CHECK: vfchdb [[REG:%v[0-9]+]], %v26, %v24
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uge <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ule selects.
define <2 x double> @f25(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f25:
; CHECK: vfchdb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ule <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ult selects.
define <2 x double> @f26(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f26:
; CHECK: vfchedb [[REG:%v[0-9]+]], %v24, %v26
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ult <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test ord selects.
define <2 x double> @f27(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f27:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfchedb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ord <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test uno selects.
define <2 x double> @f28(<2 x double> %val1, <2 x double> %val2,
                         <2 x double> %val3, <2 x double> %val4) {
; CHECK-LABEL: f28:
; CHECK-DAG: vfchdb [[REG1:%v[0-9]+]], %v26, %v24
; CHECK-DAG: vfchedb [[REG2:%v[0-9]+]], %v24, %v26
; CHECK: vo [[REG:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uno <2 x double> %val1, %val2
  %ret = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %ret
}

; Test an f64 comparison that uses vector registers.
define i64 @f29(i64 %a, i64 %b, double %f1, <2 x double> %vec) {
; CHECK-LABEL: f29:
; CHECK: wfcdb %f0, %v24
; CHECK-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f2 = extractelement <2 x double> %vec, i32 0
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
