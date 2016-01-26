; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs -o - %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -o - %s

; SI-LABEL:{{^}}row_filter_C1_D0:
; SI: s_endpgm
; Function Attrs: nounwind
define void @row_filter_C1_D0() {
entry:
  br i1 undef, label %for.inc.1, label %do.body.preheader

do.body.preheader:                                ; preds = %entry
  %0 = insertelement <4 x i32> zeroinitializer, i32 undef, i32 1
  br i1 undef, label %do.body56.1, label %do.body90

do.body90:                                        ; preds = %do.body56.2, %do.body56.1, %do.body.preheader
  %1 = phi <4 x i32> [ %6, %do.body56.2 ], [ %5, %do.body56.1 ], [ %0, %do.body.preheader ]
  %2 = insertelement <4 x i32> %1, i32 undef, i32 2
  %3 = insertelement <4 x i32> %2, i32 undef, i32 3
  br i1 undef, label %do.body124.1, label %do.body.1562.preheader

do.body.1562.preheader:                           ; preds = %do.body124.1, %do.body90
  %storemerge = phi <4 x i32> [ %3, %do.body90 ], [ %7, %do.body124.1 ]
  %4 = insertelement <4 x i32> undef, i32 undef, i32 1
  br label %for.inc.1

do.body56.1:                                      ; preds = %do.body.preheader
  %5 = insertelement <4 x i32> %0, i32 undef, i32 1
  %or.cond472.1 = or i1 undef, undef
  br i1 %or.cond472.1, label %do.body56.2, label %do.body90

do.body56.2:                                      ; preds = %do.body56.1
  %6 = insertelement <4 x i32> %5, i32 undef, i32 1
  br label %do.body90

do.body124.1:                                     ; preds = %do.body90
  %7 = insertelement <4 x i32> %3, i32 undef, i32 3
  br label %do.body.1562.preheader

for.inc.1:                                        ; preds = %do.body.1562.preheader, %entry
  %storemerge591 = phi <4 x i32> [ zeroinitializer, %entry ], [ %storemerge, %do.body.1562.preheader ]
  %add.i495 = add <4 x i32> %storemerge591, undef
  unreachable
}

; SI-LABEL: {{^}}foo:
; SI: s_endpgm
define void @foo() #0 {
bb:
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb4, label %bb6

bb2:                                              ; preds = %bb4, %bb
  %tmp = phi float [ %tmp5, %bb4 ], [ 0.000000e+00, %bb ]
  br i1 undef, label %bb9, label %bb13

bb4:                                              ; preds = %bb7, %bb6, %bb1
  %tmp5 = phi float [ undef, %bb1 ], [ undef, %bb6 ], [ %tmp8, %bb7 ]
  br label %bb2

bb6:                                              ; preds = %bb1
  br i1 undef, label %bb7, label %bb4

bb7:                                              ; preds = %bb6
  %tmp8 = fmul float undef, undef
  br label %bb4

bb9:                                              ; preds = %bb2
  %tmp10 = call <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp11 = extractelement <4 x float> %tmp10, i32 1
  %tmp12 = extractelement <4 x float> %tmp10, i32 3
  br label %bb14

bb13:                                             ; preds = %bb2
  br i1 undef, label %bb23, label %bb24

bb14:                                             ; preds = %bb27, %bb24, %bb9
  %tmp15 = phi float [ %tmp12, %bb9 ], [ undef, %bb27 ], [ 0.000000e+00, %bb24 ]
  %tmp16 = phi float [ %tmp11, %bb9 ], [ undef, %bb27 ], [ %tmp25, %bb24 ]
  %tmp17 = fmul float 10.5, %tmp16
  %tmp18 = fmul float 11.5, %tmp15
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %tmp18, float %tmp17, float %tmp17, float %tmp17)
  ret void

bb23:                                             ; preds = %bb13
  br i1 undef, label %bb24, label %bb26

bb24:                                             ; preds = %bb26, %bb23, %bb13
  %tmp25 = phi float [ %tmp, %bb13 ], [ %tmp, %bb26 ], [ 0.000000e+00, %bb23 ]
  br i1 undef, label %bb27, label %bb14

bb26:                                             ; preds = %bb23
  br label %bb24

bb27:                                             ; preds = %bb24
  br label %bb14
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.image.sample.v2i32(<2 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" "enable-no-nans-fp-math"="true" "unsafe-fp-math"="true" }
attributes #1 = { nounwind readnone }
