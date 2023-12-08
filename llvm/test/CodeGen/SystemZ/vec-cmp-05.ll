; Test v4f32 comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test oeq.
define <4 x i32> @f1(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f1:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfcedb [[HIGHRES:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfcedb [[LOWRES:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK: vpkg %v24, [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp oeq <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test one.
define <4 x i32> @f2(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f2:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchdb [[HIGHRES0:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchdb [[LOWRES0:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK-DAG: vfchdb [[HIGHRES1:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchdb [[LOWRES1:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK-DAG: vpkg [[RES0:%v[0-9]+]], [[HIGHRES0]], [[LOWRES0]]
; CHECK-DAG: vpkg [[RES1:%v[0-9]+]], [[HIGHRES1]], [[LOWRES1]]
; CHECK: vo %v24, [[RES1]], [[RES0]]
; CHECK-NEXT: br %r14
  %cmp = fcmp one <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ogt.
define <4 x i32> @f3(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f3:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchdb [[HIGHRES:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchdb [[LOWRES:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK: vpkg %v24, [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test oge.
define <4 x i32> @f4(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f4:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchedb [[HIGHRES:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchedb [[LOWRES:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK: vpkg %v24, [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp oge <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ole.
define <4 x i32> @f5(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f5:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchedb [[HIGHRES:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchedb [[LOWRES:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK: vpkg %v24, [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ole <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test olt.
define <4 x i32> @f6(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f6:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchdb [[HIGHRES:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchdb [[LOWRES:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK: vpkg %v24, [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp olt <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ueq.
define <4 x i32> @f7(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f7:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchdb [[HIGHRES0:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchdb [[LOWRES0:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK-DAG: vfchdb [[HIGHRES1:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchdb [[LOWRES1:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK-DAG: vpkg [[RES0:%v[0-9]+]], [[HIGHRES0]], [[LOWRES0]]
; CHECK-DAG: vpkg [[RES1:%v[0-9]+]], [[HIGHRES1]], [[LOWRES1]]
; CHECK: vno %v24, [[RES1]], [[RES0]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ueq <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test une.
define <4 x i32> @f8(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f8:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfcedb [[HIGHRES:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfcedb [[LOWRES:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK: vpkg [[RES:%v[0-9]+]], [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: vno %v24, [[RES]], [[RES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp une <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ugt.
define <4 x i32> @f9(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f9:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchedb [[HIGHRES:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchedb [[LOWRES:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK: vpkg [[RES:%v[0-9]+]], [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: vno %v24, [[RES]], [[RES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ugt <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test uge.
define <4 x i32> @f10(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f10:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchdb [[HIGHRES:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchdb [[LOWRES:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK: vpkg [[RES:%v[0-9]+]], [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: vno %v24, [[RES]], [[RES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uge <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ule.
define <4 x i32> @f11(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f11:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchdb [[HIGHRES:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchdb [[LOWRES:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK: vpkg [[RES:%v[0-9]+]], [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: vno %v24, [[RES]], [[RES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ule <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ult.
define <4 x i32> @f12(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f12:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchedb [[HIGHRES:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchedb [[LOWRES:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK: vpkg [[RES:%v[0-9]+]], [[HIGHRES]], [[LOWRES]]
; CHECK-NEXT: vno %v24, [[RES]], [[RES]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ult <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test ord.
define <4 x i32> @f13(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f13:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchedb [[HIGHRES0:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchedb [[LOWRES0:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK-DAG: vfchdb [[HIGHRES1:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchdb [[LOWRES1:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK-DAG: vpkg [[RES0:%v[0-9]+]], [[HIGHRES0]], [[LOWRES0]]
; CHECK-DAG: vpkg [[RES1:%v[0-9]+]], [[HIGHRES1]], [[LOWRES1]]
; CHECK: vo %v24, [[RES1]], [[RES0]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ord <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test uno.
define <4 x i32> @f14(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f14:
; CHECK-DAG: vmrhf [[HIGH0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrlf [[LOW0E:%v[0-9]+]], %v24, %v24
; CHECK-DAG: vmrhf [[HIGH1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vmrlf [[LOW1E:%v[0-9]+]], %v26, %v26
; CHECK-DAG: vldeb [[HIGH0D:%v[0-9]+]], [[HIGH0E]]
; CHECK-DAG: vldeb [[HIGH1D:%v[0-9]+]], [[HIGH1E]]
; CHECK-DAG: vldeb [[LOW0D:%v[0-9]+]], [[LOW0E]]
; CHECK-DAG: vldeb [[LOW1D:%v[0-9]+]], [[LOW1E]]
; CHECK-DAG: vfchedb [[HIGHRES0:%v[0-9]+]], [[HIGH0D]], [[HIGH1D]]
; CHECK-DAG: vfchedb [[LOWRES0:%v[0-9]+]], [[LOW0D]], [[LOW1D]]
; CHECK-DAG: vfchdb [[HIGHRES1:%v[0-9]+]], [[HIGH1D]], [[HIGH0D]]
; CHECK-DAG: vfchdb [[LOWRES1:%v[0-9]+]], [[LOW1D]], [[LOW0D]]
; CHECK-DAG: vpkg [[RES0:%v[0-9]+]], [[HIGHRES0]], [[LOWRES0]]
; CHECK-DAG: vpkg [[RES1:%v[0-9]+]], [[HIGHRES1]], [[LOWRES1]]
; CHECK: vno %v24, [[RES1]], [[RES0]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uno <4 x float> %val1, %val2
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test oeq selects.
define <4 x float> @f15(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f15:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp oeq <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test one selects.
define <4 x float> @f16(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f16:
; CHECK: vo [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp one <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ogt selects.
define <4 x float> @f17(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f17:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test oge selects.
define <4 x float> @f18(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f18:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp oge <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ole selects.
define <4 x float> @f19(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f19:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ole <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test olt selects.
define <4 x float> @f20(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f20:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp olt <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ueq selects.
define <4 x float> @f21(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f21:
; CHECK: vo [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ueq <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test une selects.
define <4 x float> @f22(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f22:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp une <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ugt selects.
define <4 x float> @f23(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f23:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ugt <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test uge selects.
define <4 x float> @f24(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f24:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uge <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ule selects.
define <4 x float> @f25(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f25:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ule <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ult selects.
define <4 x float> @f26(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f26:
; CHECK: vpkg [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ult <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test ord selects.
define <4 x float> @f27(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f27:
; CHECK: vo [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v28, %v30, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp ord <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}

; Test uno selects.
define <4 x float> @f28(<4 x float> %val1, <4 x float> %val2,
                        <4 x float> %val3, <4 x float> %val4) {
; CHECK-LABEL: f28:
; CHECK: vo [[REG:%v[0-9]+]],
; CHECK-NEXT: vsel %v24, %v30, %v28, [[REG]]
; CHECK-NEXT: br %r14
  %cmp = fcmp uno <4 x float> %val1, %val2
  %ret = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %ret
}
