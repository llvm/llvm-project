; RUN: opt %loadNPMPolly -passes=polly-codegen -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; CHECK:      entry:
; CHECK-NEXT:   %outvalue.141.phiops = alloca i64
; CHECK-NEXT:   %.preload.s2a = alloca i8
; CHECK-NEXT:   %umax = call i32 @llvm.umax.i32(i32 undef, i32 1)
; CHECK-NEXT:   %divpolly = sdiv i32 undef, %umax
; CHECK-NEXT:   %div = sdiv i32 undef, undef
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @int_downsample() #0 {
entry:
  %div = sdiv i32 undef, undef
  br label %for.cond10.preheader.lr.ph

for.cond10.preheader.lr.ph:                       ; preds = %entry
  br label %for.body17.lr.ph

for.body17.lr.ph:                                 ; preds = %for.end22, %for.cond10.preheader.lr.ph
  %outcol_h.048 = phi i32 [ 0, %for.cond10.preheader.lr.ph ], [ %add31, %for.end22 ]
  %0 = load ptr, ptr undef
  %idx.ext = zext i32 %outcol_h.048 to i64
  %add.ptr = getelementptr inbounds i8, ptr %0, i64 %idx.ext
  br label %for.body17

for.body17:                                       ; preds = %for.body17, %for.body17.lr.ph
  %outvalue.141 = phi i64 [ poison, %for.body17.lr.ph ], [ %add19, %for.body17 ]
  %inptr.040 = phi ptr [ %add.ptr, %for.body17.lr.ph ], [ poison, %for.body17 ]
  %1 = load i8, ptr %inptr.040
  %add19 = mul nsw i64 0, %outvalue.141
  br i1 false, label %for.body17, label %for.end22

for.end22:                                        ; preds = %for.body17
  %add31 = add i32 %outcol_h.048, %div
  br i1 undef, label %for.body17.lr.ph, label %for.end32

for.end32:                                        ; preds = %for.end22
  br label %for.end36

for.end36:                                        ; preds = %for.end32
  ret void
}
