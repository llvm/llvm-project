; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define float @extract_from_zero_init_shuffle(<2 x float> %1, i64 %idx) {
; CHECK-LABEL: @extract_from_zero_init_shuffle(
; CHECK-NEXT:    [[TMP1:%.*]] = extractelement <2 x float> [[W:%.*]], i64 0
; CHECK-NEXT:    ret float [[TMP1]]
;
  %3 = shufflevector <2 x float> %1, <2 x float> poison, <4 x i32> zeroinitializer
  %4 = extractelement <4 x float> %3, i64 %idx
  ret float %4
}

