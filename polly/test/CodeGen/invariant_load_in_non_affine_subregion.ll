; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-invariant-load-hoisting=true -S < %s | FileCheck %s
;
; This crashed at some point as the invariant load is in a non-affine
; subregion. Just check it does not anymore.
;
; CHECK: polly.start
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.d = type { i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@board = external global [421 x i8], align 16
@output_flags = external global i32, align 4
@dragon = external global [400 x %struct.d], align 16

; Function Attrs: nounwind uwtable
define void @sgffile_add_debuginfo() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %0 = getelementptr inbounds [24 x i8], [24 x i8]* undef, i64 0, i64 0
  br i1 false, label %cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry.split
  br i1 false, label %if.then7, label %land.lhs.true49

if.then7:                                         ; preds = %for.cond1.preheader
  %arrayidx = getelementptr inbounds [421 x i8], [421 x i8]* @board, i64 0, i64 0
  %crude_status = getelementptr inbounds [400 x %struct.d], [400 x %struct.d]* @dragon, i64 0, i64 0, i32 5
  switch i32 0, label %if.end15 [
    i32 0, label %sw.bb
    i32 2, label %sw.bb13
  ]

sw.bb:                                            ; preds = %if.then7
  br label %if.end15

sw.bb13:                                          ; preds = %if.then7
  br label %if.end15

if.end15:                                         ; preds = %sw.bb13, %sw.bb, %if.then7
  %cmp21 = fcmp ogt float undef, 0.000000e+00
  br i1 %cmp21, label %land.lhs.true23, label %for.cond1.for.inc44_crit_edge

land.lhs.true23:                                  ; preds = %if.end15
  %1 = load i32, i32* @output_flags, align 4
  %and24 = and i32 %1, 2
  %tobool25 = icmp eq i32 %and24, 0
  br i1 %tobool25, label %for.cond1.for.inc44_crit_edge, label %if.else

if.else:                                          ; preds = %land.lhs.true23
  br label %for.cond1.for.inc44_crit_edge

for.cond1.for.inc44_crit_edge:                    ; preds = %if.else, %land.lhs.true23, %if.end15
  br label %land.lhs.true49

land.lhs.true49:                                  ; preds = %for.cond1.for.inc44_crit_edge, %for.cond1.preheader
  %2 = load i32, i32* @output_flags, align 4
  %and50 = and i32 %2, 2
  %tobool51 = icmp eq i32 %and50, 0
  br i1 %tobool51, label %cleanup, label %if.then52

if.then52:                                        ; preds = %land.lhs.true49
  br label %cleanup

cleanup:                                          ; preds = %if.then52, %land.lhs.true49, %entry.split
  call void @dummy(ptr %0)
  ret void
}

declare void @dummy(ptr)
