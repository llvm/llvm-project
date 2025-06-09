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


define float @extract_from_general_splat(<2 x float> %1, i64 %idx) {
; CHECK-LABEL: @extract_from_general_splat(
; CHECK-NEXT:    [[TMP1:%.*]] = extractelement <2 x float> [[W:%.*]], i64 1
; CHECK-NEXT:    ret float [[TMP1]]
;
  %3 = shufflevector <2 x float> %1, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %4 = extractelement <4 x float> %3, i64 %idx
  ret float %4
}

define float @extract_from_general_scalable_splat(<vscale x 2 x float> %1, i64 %idx) {
; CHECK-LABEL: @extract_from_general_scalable_splat(
; CHECK-NEXT:    [[TMP1:%.*]] = extractelement <vscale x 2 x float> [[W:%.*]], i64 0
; CHECK-NEXT:    ret float [[TMP1]]
;
  %3 = shufflevector <vscale x 2 x float> %1, <vscale x 2 x float> poison, <vscale x 4 x i32> zeroinitializer
  %4 = extractelement <vscale x 4 x float> %3, i64 %idx
  ret float %4
}

define float @no_extract_from_general_no_splat_0(<2 x float> %1, i64 %idx) {
; CHECK-LABEL: @no_extract_from_general_no_splat_0(
; CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <2 x float> [[W:%.*]], <2 x float> poison, <4 x i32> <i32 poison, i32 1, i32 1, i32 1>
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x float> [[TMP1]], i64 %idx
; CHECK-NEXT:    ret float [[TMP2]]
;
  %3 = shufflevector <2 x float> %1, <2 x float> poison, <4 x i32> <i32 poison, i32 1, i32 1, i32 1>
  %4 = extractelement <4 x float> %3, i64 %idx
  ret float %4
}

define float @no_extract_from_general_no_splat_1(<2 x float> %1, i64 %idx) {
; CHECK-LABEL: @no_extract_from_general_no_splat_1(
; CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <2 x float> [[W:%.*]], <2 x float> poison, <4 x i32> <i32 1, i32 poison, i32 1, i32 1>
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x float> [[TMP1]], i64 %idx
; CHECK-NEXT:    ret float [[TMP2]]
;
  %3 = shufflevector <2 x float> %1, <2 x float> poison, <4 x i32> <i32 1, i32 poison, i32 1, i32 1>
  %4 = extractelement <4 x float> %3, i64 %idx
  ret float %4
}
