; RUN: opt %loadNPMPolly "-passes=scop(print<polly-opt-isl>)" -polly-vectorizer=stripmine -polly-invariant-load-hoisting -disable-output < %s | FileCheck %s
;
; llvm.org/PR46578
;
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

%struct.stCfg_Incell_DetailThreshold.2.30.42.62.74.94.122.126.134.166.194.242.338.342.346.350.354.358.496.0.2.9.16.28.36.37.38.39.40.75 = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr }
@ft8006m_g_stCfg_Incell_DetailThreshold = external dso_local local_unnamed_addr global %struct.stCfg_Incell_DetailThreshold.2.30.42.62.74.94.122.126.134.166.194.242.338.342.346.350.354.358.496.0.2.9.16.28.36.37.38.39.40.75, align 8
declare dso_local i32 @ft8006m_atoi() local_unnamed_addr #0

define void @func() {
entry:
  switch i32 undef, label %cleanup [
    i32 10, label %if.end
    i32 14, label %if.end
    i32 16, label %if.end
  ]

if.end:                                           ; preds = %entry, %entry, %entry
  %call15 = call i32 @ft8006m_atoi() #1
  %0 = zext i32 %call15 to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc39, %if.end
  %indvars.iv302 = phi i64 [ %indvars.iv.next303, %for.inc39 ], [ 0, %if.end ]
  %exitcond304 = icmp eq i64 %indvars.iv302, 60
  br i1 %exitcond304, label %cleanup, label %for.cond21

for.cond21:                                       ; preds = %for.body23, %for.cond
  %indvars.iv296 = phi i64 [ %indvars.iv.next297, %for.body23 ], [ 0, %for.cond ]
  %exitcond298 = icmp eq i64 %indvars.iv296, 60
  br i1 %exitcond298, label %for.cond28, label %for.body23

for.body23:                                       ; preds = %for.cond21
  %1 = load ptr, ptr getelementptr inbounds (%struct.stCfg_Incell_DetailThreshold.2.30.42.62.74.94.122.126.134.166.194.242.338.342.346.350.354.358.496.0.2.9.16.28.36.37.38.39.40.75, ptr @ft8006m_g_stCfg_Incell_DetailThreshold, i64 0, i32 2), align 8
  %arrayidx25 = getelementptr [60 x i32], ptr %1, i64 %indvars.iv302, i64 %indvars.iv296
  store i32 undef, ptr %arrayidx25, align 4
  %indvars.iv.next297 = add nuw nsw i64 %indvars.iv296, 1
  br label %for.cond21

for.cond28:                                       ; preds = %for.body30, %for.cond21
  %indvars.iv299 = phi i64 [ %indvars.iv.next300, %for.body30 ], [ 0, %for.cond21 ]
  %exitcond301 = icmp eq i64 %indvars.iv299, 60
  br i1 %exitcond301, label %for.inc39, label %for.body30

for.body30:                                       ; preds = %for.cond28
  %2 = load ptr, ptr getelementptr inbounds (%struct.stCfg_Incell_DetailThreshold.2.30.42.62.74.94.122.126.134.166.194.242.338.342.346.350.354.358.496.0.2.9.16.28.36.37.38.39.40.75, ptr @ft8006m_g_stCfg_Incell_DetailThreshold, i64 0, i32 2), align 8
  %arrayidx34 = getelementptr [60 x i32], ptr %2, i64 %0, i64 %indvars.iv299
  store i32 undef, ptr %arrayidx34, align 4
  %indvars.iv.next300 = add nuw nsw i64 %indvars.iv299, 1
  br label %for.cond28

for.inc39:                                        ; preds = %for.cond28
  %indvars.iv.next303 = add nuw nsw i64 %indvars.iv302, 1
  br label %for.cond

cleanup:                                          ; preds = %for.cond, %entry
  ret void
}


; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for.cond => cleanup' in function 'func':
; CHECK: Calculated schedule:
; CHECK: domain: "[call15] -> { Stmt_for_body30[i0, i1] : 0 <= i0 <= 59 and 0 <= i1 <= 59; Stmt_for_body23[i0, i1] : 0 <= i0 <= 59 and 0 <= i1 <= 59 }"
; CHECK: child:
; CHECK:   mark: "1st level tiling - Tiles"
; CHECK:   child:
; CHECK:     schedule: "[call15] -> [{ Stmt_for_body30[i0, i1] -> [(floor((i1)/32))]; Stmt_for_body23[i0, i1] -> [(floor((i1)/32))] }, { Stmt_for_body30[i0, i1] -> [(floor((i0)/32))]; Stmt_for_body23[i0, i1] -> [(floor((i0)/32))] }]"
; CHECK:     permutable: 1
; CHECK:     coincident: [ 1, 0 ]
; CHECK:     child:
; CHECK:       mark: "1st level tiling - Points"
; CHECK:       child:
; CHECK:         schedule: "[call15] -> [{ Stmt_for_body30[i0, i1] -> [(floor((i1)/4) - 8*floor((i1)/32))]; Stmt_for_body23[i0, i1] -> [(floor((i1)/4) - 8*floor((i1)/32))] }]"
; CHECK:         permutable: 1
; CHECK:         coincident: [ 1 ]
; CHECK:         options: "[call15] -> { atomic[0]; isolate{{\[\[}}i0, i1] -> [i2]] : i0 >= 0 and 0 <= i1 <= 1 and 0 <= i2 <= 14 - 8i0 and i2 <= 7 }"
; CHECK:         child:
; CHECK:           schedule: "[call15] -> [{ Stmt_for_body30[i0, i1] -> [((i0) mod 32)]; Stmt_for_body23[i0, i1] -> [((i0) mod 32)] }]"
; CHECK:           permutable: 1
; CHECK:           child:
; CHECK:             sequence:
; CHECK:             - filter: "[call15] -> { Stmt_for_body23[i0, i1] }"
; CHECK:               child:
; CHECK:                 mark: "SIMD"
; CHECK:                 child:
; CHECK:                   schedule: "[call15] -> [{ Stmt_for_body30[i0, i1] -> [((i1) mod 4)]; Stmt_for_body23[i0, i1] -> [((i1) mod 4)] }]"
; CHECK:                   permutable: 1
; CHECK:                   coincident: [ 1 ]
; CHECK:             - filter: "[call15] -> { Stmt_for_body30[i0, i1] }"
; CHECK:               child:
; CHECK:                 mark: "SIMD"
; CHECK:                 child:
; CHECK:                   schedule: "[call15] -> [{ Stmt_for_body30[i0, i1] -> [((i1) mod 4)]; Stmt_for_body23[i0, i1] -> [((i1) mod 4)] }]"
; CHECK:                   permutable: 1
; CHECK:                   coincident: [ 1 ]
