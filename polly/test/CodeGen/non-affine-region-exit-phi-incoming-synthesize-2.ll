; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; This caused the code generation to generate invalid code as the same operand
; of the PHI node in the non-affine region was synthesized at the wrong place.
; Check we do not generate wrong code.
;
; CHECK: polly.start
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@connected_passed = external global [256 x i8], align 16

; Function Attrs: norecurse nounwind uwtable
define void @InitializeZeroMasks() {
entry:
  br label %for.end20

for.end20:                                        ; preds = %entry
  br label %for.body24

for.body24:                                       ; preds = %for.body61.preheader, %for.end20
  %indvars.iv = phi i64 [ 0, %for.end20 ], [ %indvars.iv.next, %for.body61.preheader ]
  %arrayidx26 = getelementptr inbounds [256 x i8], ptr @connected_passed, i64 0, i64 %indvars.iv
  store i8 0, ptr %arrayidx26, align 1
  %0 = trunc i64 %indvars.iv to i32
  br i1 false, label %for.inc56.4, label %if.then51

if.then51:                                        ; preds = %for.inc56.5, %for.body24
  %j.342.lcssa = phi i8 [ 7, %for.body24 ], [ %.mux, %for.inc56.5 ]
  br label %for.body61.preheader

for.body61.preheader:                             ; preds = %for.inc56.5, %if.then51
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end79, label %for.body24

for.end79:                                        ; preds = %for.body61.preheader
  ret void

for.inc56.4:                                      ; preds = %for.body24
  br label %for.inc56.5

for.inc56.5:                                      ; preds = %for.inc56.4
  %and49.6 = and i32 %0, 64
  %brmerge = or i1 undef, undef
  %and49.6.lobit = lshr exact i32 %and49.6, 6
  %.mux = trunc i32 %and49.6.lobit to i8
  br i1 %brmerge, label %if.then51, label %for.body61.preheader
}
