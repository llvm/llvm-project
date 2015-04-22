; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs -o - %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -o - %s
; ModuleID = 'bugpoint-reduced-simplified.bc'

; SI: s_endpgm
; Function Attrs: nounwind
define void @row_filter_C1_D0() #0 {
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

