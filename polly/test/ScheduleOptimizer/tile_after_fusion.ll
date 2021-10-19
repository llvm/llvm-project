; RUN: opt %loadPolly -polly-opt-isl -polly-ast -polly-isl-arg=--no-schedule-serialize-sccs -analyze < %s | FileCheck %s
;
;
;    void tf(int C[256][256][256], int A0[256][256][256], int A1[256][256][256]) {
;      for (int i = 0; i < 256; ++i)
;        for (int j = 0; j < 256; ++j)
;          for (int k = 0; k < 256; ++k)
;            C[i][j][k] += A0[i][j][k];
;
;      for (int i = 0; i < 256; ++i)
;        for (int j = 0; j < 256; ++j)
;          for (int k = 0; k < 256; ++k)
;            C[i][j][k] += A1[i][j][k];
;    }
;
; The tile_after_fusion.ll test has two statements in separate loop nests and
; checks whether they are tiled after being fused when polly-opt-fusion equals
; "max".
;
; CHECK-LABEL: Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: 'for.cond => for.end56' in function 'tf':
; CHECK:       1st level tiling - Tiles
; CHECK-NEXT:     for (int c0 = 0; c0 <= 7; c0 += 1)
; CHECK-NEXT:       for (int c1 = 0; c1 <= 7; c1 += 1)
; CHECK-NEXT:         for (int c2 = 0; c2 <= 7; c2 += 1) {
; CHECK-NEXT:           // 1st level tiling - Points
; CHECK-NEXT:           for (int c3 = 0; c3 <= 31; c3 += 1)
; CHECK-NEXT:             for (int c4 = 0; c4 <= 31; c4 += 1)
; CHECK-NEXT:               for (int c5 = 0; c5 <= 31; c5 += 1) {
; CHECK-NEXT:                 Stmt_for_body6(32 * c0 + c3, 32 * c1 + c4, 32 * c2 + c5);
; CHECK-NEXT:                 Stmt_for_body34(32 * c0 + c3, 32 * c1 + c4, 32 * c2 + c5);

source_filename = "tile_after_fusion.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @tf([256 x [256 x i32]]* %C, [256 x [256 x i32]]* %A0, [256 x [256 x i32]]* %A1) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc20, %entry
  %indvars.iv13 = phi i64 [ %indvars.iv.next14, %for.inc20 ], [ 0, %entry ]
  %exitcond15 = icmp ne i64 %indvars.iv13, 256
  br i1 %exitcond15, label %for.body, label %for.end22

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc17, %for.body
  %indvars.iv10 = phi i64 [ %indvars.iv.next11, %for.inc17 ], [ 0, %for.body ]
  %exitcond12 = icmp ne i64 %indvars.iv10, 256
  br i1 %exitcond12, label %for.body3, label %for.end19

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %indvars.iv7 = phi i64 [ %indvars.iv.next8, %for.inc ], [ 0, %for.body3 ]
  %exitcond9 = icmp ne i64 %indvars.iv7, 256
  br i1 %exitcond9, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %arrayidx10 = getelementptr inbounds [256 x [256 x i32]], [256 x [256 x i32]]* %A0, i64 %indvars.iv13, i64 %indvars.iv10, i64 %indvars.iv7
  %tmp = load i32, i32* %arrayidx10, align 4
  %arrayidx16 = getelementptr inbounds [256 x [256 x i32]], [256 x [256 x i32]]* %C, i64 %indvars.iv13, i64 %indvars.iv10, i64 %indvars.iv7
  %tmp16 = load i32, i32* %arrayidx16, align 4
  %add = add nsw i32 %tmp16, %tmp
  store i32 %add, i32* %arrayidx16, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc17

for.inc17:                                        ; preds = %for.end
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  br label %for.cond1

for.end19:                                        ; preds = %for.cond1
  br label %for.inc20

for.inc20:                                        ; preds = %for.end19
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 1
  br label %for.cond

for.end22:                                        ; preds = %for.cond
  br label %for.cond24

for.cond24:                                       ; preds = %for.inc54, %for.end22
  %indvars.iv4 = phi i64 [ %indvars.iv.next5, %for.inc54 ], [ 0, %for.end22 ]
  %exitcond6 = icmp ne i64 %indvars.iv4, 256
  br i1 %exitcond6, label %for.body26, label %for.end56

for.body26:                                       ; preds = %for.cond24
  br label %for.cond28

for.cond28:                                       ; preds = %for.inc51, %for.body26
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc51 ], [ 0, %for.body26 ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 256
  br i1 %exitcond3, label %for.body30, label %for.end53

for.body30:                                       ; preds = %for.cond28
  br label %for.cond32

for.cond32:                                       ; preds = %for.inc48, %for.body30
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc48 ], [ 0, %for.body30 ]
  %exitcond = icmp ne i64 %indvars.iv, 256
  br i1 %exitcond, label %for.body34, label %for.end50

for.body34:                                       ; preds = %for.cond32
  %arrayidx40 = getelementptr inbounds [256 x [256 x i32]], [256 x [256 x i32]]* %A1, i64 %indvars.iv4, i64 %indvars.iv1, i64 %indvars.iv
  %tmp17 = load i32, i32* %arrayidx40, align 4
  %arrayidx46 = getelementptr inbounds [256 x [256 x i32]], [256 x [256 x i32]]* %C, i64 %indvars.iv4, i64 %indvars.iv1, i64 %indvars.iv
  %tmp18 = load i32, i32* %arrayidx46, align 4
  %add47 = add nsw i32 %tmp18, %tmp17
  store i32 %add47, i32* %arrayidx46, align 4
  br label %for.inc48

for.inc48:                                        ; preds = %for.body34
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond32

for.end50:                                        ; preds = %for.cond32
  br label %for.inc51

for.inc51:                                        ; preds = %for.end50
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond28

for.end53:                                        ; preds = %for.cond28
  br label %for.inc54

for.inc54:                                        ; preds = %for.end53
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1
  br label %for.cond24

for.end56:                                        ; preds = %for.cond24
  ret void
}
