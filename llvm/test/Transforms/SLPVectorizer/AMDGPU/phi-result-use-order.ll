; RUN: opt -passes=slp-vectorizer -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck %s

define <4 x half> @phis(i1 %cmp1, <4 x half> %in1, <4 x half> %in2)  {
; CHECK-LABEL: @phis(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A0:%.*]] = extractelement <4 x half> [[IN1:%.*]], i64 0
; CHECK-NEXT:    [[A1:%.*]] = extractelement <4 x half> [[IN1]], i64 1
; CHECK-NEXT:    [[A2:%.*]] = extractelement <4 x half> [[IN1]], i64 2
; CHECK-NEXT:    [[A3:%.*]] = extractelement <4 x half> [[IN1]], i64 3
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x half> poison, half [[A0]], i32 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <2 x half> [[TMP0]], half [[A1]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x half> poison, half [[A2]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x half> [[TMP2]], half [[A3]], i32 1
; CHECK-NEXT:    br i1 [[CMP:%.*]], label [[BB1:%.*]], label [[BB0:%.*]]
; CHECK:       bb0:
; CHECK-NEXT:    [[B0:%.*]] = extractelement <4 x half> [[IN2:%.*]], i64 0
; CHECK-NEXT:    [[B1:%.*]] = extractelement <4 x half> [[IN2]], i64 1
; CHECK-NEXT:    [[B2:%.*]] = extractelement <4 x half> [[IN2]], i64 2
; CHECK-NEXT:    [[B3:%.*]] = extractelement <4 x half> [[IN2]], i64 3
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <2 x half> poison, half [[B0]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <2 x half> [[TMP4]], half [[B1]], i32 1
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <2 x half> poison, half [[B2]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = insertelement <2 x half> [[TMP6]], half [[B3]], i32 1
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP8:%.*]] = phi <2 x half> [ [[TMP1]], %entry ], [ [[TMP5]], %bb0 ]
; CHECK-NEXT:    [[TMP9:%.*]] = phi <2 x half> [ [[TMP3]], %entry ], [ [[TMP7]], %bb0 ]
; CHECK-NEXT:    [[TMP10:%.*]] = shufflevector <2 x half> [[TMP8]], <2 x half> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT:    [[TMP11:%.*]] = shufflevector <2 x half> [[TMP9]], <2 x half> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT:    [[TMP12:%.*]] = shufflevector <4 x half> [[TMP10]], <4 x half> [[TMP11]], <4 x i32> <i32 0, i32 1, i32 4, i32 5>
; CHECK-NEXT:    ret <4 x half> [[TMP12]]
entry:
  %a0 = extractelement <4 x half> %in1, i64 0
  %a1 = extractelement <4 x half> %in1, i64 1
  %a2 = extractelement <4 x half> %in1, i64 2
  %a3 = extractelement <4 x half> %in1, i64 3
  br i1 %cmp1, label %bb1, label %bb0

bb0:
  %b0 = extractelement <4 x half> %in2, i64 0
  %b1 = extractelement <4 x half> %in2, i64 1
  %b2 = extractelement <4 x half> %in2, i64 2
  %b3 = extractelement <4 x half> %in2, i64 3
  br label %bb1

bb1:
  %c0 = phi half [ %a0, %entry ], [ %b0, %bb0 ]
  %c1 = phi half [ %a1, %entry ], [ %b1, %bb0 ]
  %c2 = phi half [ %a2, %entry ], [ %b2, %bb0 ]
  %c3 = phi half [ %a3, %entry ], [ %b3, %bb0 ]

  %o0 = insertelement <4 x half> undef, half %c0, i64 0
  %o1 = insertelement <4 x half> %o0, half %c1, i64 1
  %o2 = insertelement <4 x half> %o1, half %c2, i64 2
  %o3 = insertelement <4 x half> %o2, half %c3, i64 3
  ret <4 x half> %o3
}

define <4 x half> @phis_reverse(i1 %cmp1, <4 x half> %in1, <4 x half> %in2)  {
; CHECK-LABEL: @phis_reverse(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A0:%.*]] = extractelement <4 x half> [[IN1:%.*]], i64 0
; CHECK-NEXT:    [[A1:%.*]] = extractelement <4 x half> [[IN1]], i64 1
; CHECK-NEXT:    [[A2:%.*]] = extractelement <4 x half> [[IN1]], i64 2
; CHECK-NEXT:    [[A3:%.*]] = extractelement <4 x half> [[IN1]], i64 3
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x half> poison, half [[A0]], i32 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <2 x half> [[TMP0]], half [[A1]], i32 1
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x half> poison, half [[A2]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x half> [[TMP2]], half [[A3]], i32 1
; CHECK-NEXT:    br i1 [[CMP:%.*]], label [[BB1:%.*]], label [[BB0:%.*]]
; CHECK:       bb0:
; CHECK-NEXT:    [[B0:%.*]] = extractelement <4 x half> [[IN2:%.*]], i64 0
; CHECK-NEXT:    [[B1:%.*]] = extractelement <4 x half> [[IN2]], i64 1
; CHECK-NEXT:    [[B2:%.*]] = extractelement <4 x half> [[IN2]], i64 2
; CHECK-NEXT:    [[B3:%.*]] = extractelement <4 x half> [[IN2]], i64 3
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <2 x half> poison, half [[B0]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <2 x half> [[TMP4]], half [[B1]], i32 1
; CHECK-NEXT:    [[TMP6:%.*]] = insertelement <2 x half> poison, half [[B2]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = insertelement <2 x half> [[TMP6]], half [[B3]], i32 1
; CHECK-NEXT:    br label [[BB1:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP8:%.*]] = phi <2 x half> [ [[TMP1]], %entry ], [ [[TMP5]], %bb0 ]
; CHECK-NEXT:    [[TMP9:%.*]] = phi <2 x half> [ [[TMP3]], %entry ], [ [[TMP7]], %bb0 ]
; CHECK-NEXT:    [[TMP10:%.*]] = shufflevector <2 x half> [[TMP8]], <2 x half> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT:    [[TMP11:%.*]] = shufflevector <2 x half> [[TMP9]], <2 x half> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT:    [[TMP12:%.*]] = shufflevector <4 x half> [[TMP10]], <4 x half> [[TMP11]], <4 x i32> <i32 0, i32 1, i32 4, i32 5>
; CHECK-NEXT:    ret <4 x half> [[TMP12]]
entry:
  %a0 = extractelement <4 x half> %in1, i64 0
  %a1 = extractelement <4 x half> %in1, i64 1
  %a2 = extractelement <4 x half> %in1, i64 2
  %a3 = extractelement <4 x half> %in1, i64 3
  br i1 %cmp1, label %bb1, label %bb0

bb0:
  %b0 = extractelement <4 x half> %in2, i64 0
  %b1 = extractelement <4 x half> %in2, i64 1
  %b2 = extractelement <4 x half> %in2, i64 2
  %b3 = extractelement <4 x half> %in2, i64 3
  br label %bb1

bb1:
  %c3 = phi half [ %a3, %entry ], [ %b3, %bb0 ]
  %c2 = phi half [ %a2, %entry ], [ %b2, %bb0 ]
  %c1 = phi half [ %a1, %entry ], [ %b1, %bb0 ]
  %c0 = phi half [ %a0, %entry ], [ %b0, %bb0 ]

  %o0 = insertelement <4 x half> undef, half %c0, i64 0
  %o1 = insertelement <4 x half> %o0, half %c1, i64 1
  %o2 = insertelement <4 x half> %o1, half %c2, i64 2
  %o3 = insertelement <4 x half> %o2, half %c3, i64 3
  ret <4 x half> %o3
}
