; RUN: opt -passes=instcombine -S < %s | FileCheck %s

define float @extract_from_zero_init_shuffle(<2 x float> %vec, i64 %idx) {
; CHECK-LABEL: @extract_from_zero_init_shuffle(
; CHECK-NEXT:    %extract = extractelement <2 x float> %vec, i64 0
; CHECK-NEXT:    ret float %extract
;
  %shuffle = shufflevector <2 x float> %vec, <2 x float> poison, <4 x i32> zeroinitializer
  %extract = extractelement <4 x float> %shuffle, i64 %idx
  ret float %extract
}


define float @extract_from_general_splat(<2 x float> %vec, i64 %idx) {
; CHECK-LABEL: @extract_from_general_splat(
; CHECK-NEXT:    %extract = extractelement <2 x float> %vec, i64 1
; CHECK-NEXT:    ret float %extract
;
  %shuffle = shufflevector <2 x float> %vec, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %extract = extractelement <4 x float> %shuffle, i64 %idx
  ret float %extract
}

define float @extract_from_general_scalable_splat(<vscale x 2 x float> %vec, i64 %idx) {
; CHECK-LABEL: @extract_from_general_scalable_splat(
; CHECK-NEXT:    %extract = extractelement <vscale x 2 x float> %vec, i64 0
; CHECK-NEXT:    ret float %extract
;
  %shuffle = shufflevector <vscale x 2 x float> %vec, <vscale x 2 x float> poison, <vscale x 4 x i32> zeroinitializer
  %extract = extractelement <vscale x 4 x float> %shuffle, i64 %idx
  ret float %extract
}

define float @extract_from_splat_with_poison_0(<2 x float> %vec, i64 %idx) {
; CHECK-LABEL: @extract_from_splat_with_poison_0(
; CHECK-NEXT:    %extract = extractelement <2 x float> %vec, i64 1
; CHECK-NEXT:    ret float %extract
;
  %shuffle = shufflevector <2 x float> %vec, <2 x float> poison, <4 x i32> <i32 poison, i32 1, i32 1, i32 1>
  %extract = extractelement <4 x float> %shuffle, i64 %idx
  ret float %extract
}

define float @extract_from_splat_with_poison_1(<2 x float> %vec, i64 %idx) {
; CHECK-LABEL: @extract_from_splat_with_poison_1(
; CHECK-NEXT:    %extract = extractelement <2 x float> %vec, i64 1
; CHECK-NEXT:    ret float %extract
;
  %shuffle = shufflevector <2 x float> %vec, <2 x float> poison, <4 x i32> <i32 1, i32 poison, i32 1, i32 1>
  %extract = extractelement <4 x float> %shuffle, i64 %idx
  ret float %extract
}
