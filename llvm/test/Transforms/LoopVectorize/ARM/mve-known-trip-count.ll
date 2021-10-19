; RUN: opt -loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; Trip count of 5 - shouldn't be vectorized.
; CHECK-LABEL: tripcount5
; CHECK: LV: Selecting VF: 1
define void @tripcount5(i16* nocapture readonly %in, i32* nocapture %out, i16* nocapture readonly %consts, i32 %n) #0 {
entry:
  %arrayidx20 = getelementptr inbounds i32, i32* %out, i32 1
  %arrayidx38 = getelementptr inbounds i32, i32* %out, i32 2
  %arrayidx56 = getelementptr inbounds i32, i32* %out, i32 3
  %arrayidx74 = getelementptr inbounds i32, i32* %out, i32 4
  %arrayidx92 = getelementptr inbounds i32, i32* %out, i32 5
  %arrayidx110 = getelementptr inbounds i32, i32* %out, i32 6
  %arrayidx128 = getelementptr inbounds i32, i32* %out, i32 7
  %out.promoted = load i32, i32* %out, align 4
  %arrayidx20.promoted = load i32, i32* %arrayidx20, align 4
  %arrayidx38.promoted = load i32, i32* %arrayidx38, align 4
  %arrayidx56.promoted = load i32, i32* %arrayidx56, align 4
  %arrayidx74.promoted = load i32, i32* %arrayidx74, align 4
  %arrayidx92.promoted = load i32, i32* %arrayidx92, align 4
  %arrayidx110.promoted = load i32, i32* %arrayidx110, align 4
  %arrayidx128.promoted = load i32, i32* %arrayidx128, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store i32 %add12, i32* %out, align 4
  store i32 %add30, i32* %arrayidx20, align 4
  store i32 %add48, i32* %arrayidx38, align 4
  store i32 %add66, i32* %arrayidx56, align 4
  store i32 %add84, i32* %arrayidx74, align 4
  store i32 %add102, i32* %arrayidx92, align 4
  store i32 %add120, i32* %arrayidx110, align 4
  store i32 %add138, i32* %arrayidx128, align 4
  ret void

for.body:                                         ; preds = %entry, %for.body
  %hop.0236 = phi i32 [ 0, %entry ], [ %add139, %for.body ]
  %add12220235 = phi i32 [ %out.promoted, %entry ], [ %add12, %for.body ]
  %add30221234 = phi i32 [ %arrayidx20.promoted, %entry ], [ %add30, %for.body ]
  %add48222233 = phi i32 [ %arrayidx38.promoted, %entry ], [ %add48, %for.body ]
  %add66223232 = phi i32 [ %arrayidx56.promoted, %entry ], [ %add66, %for.body ]
  %add84224231 = phi i32 [ %arrayidx74.promoted, %entry ], [ %add84, %for.body ]
  %add102225230 = phi i32 [ %arrayidx92.promoted, %entry ], [ %add102, %for.body ]
  %add120226229 = phi i32 [ %arrayidx110.promoted, %entry ], [ %add120, %for.body ]
  %add138227228 = phi i32 [ %arrayidx128.promoted, %entry ], [ %add138, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %in, i32 %hop.0236
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, i16* %consts, i32 %hop.0236
  %1 = load i16, i16* %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %add12220235
  %add4 = or i32 %hop.0236, 1
  %arrayidx5 = getelementptr inbounds i16, i16* %in, i32 %add4
  %2 = load i16, i16* %arrayidx5, align 2
  %conv6 = sext i16 %2 to i32
  %arrayidx8 = getelementptr inbounds i16, i16* %consts, i32 %add4
  %3 = load i16, i16* %arrayidx8, align 2
  %conv9 = sext i16 %3 to i32
  %mul10 = mul nsw i32 %conv9, %conv6
  %add12 = add nsw i32 %mul10, %add
  %add13 = or i32 %hop.0236, 2
  %arrayidx14 = getelementptr inbounds i16, i16* %in, i32 %add13
  %4 = load i16, i16* %arrayidx14, align 2
  %conv15 = sext i16 %4 to i32
  %arrayidx17 = getelementptr inbounds i16, i16* %consts, i32 %add13
  %5 = load i16, i16* %arrayidx17, align 2
  %conv18 = sext i16 %5 to i32
  %mul19 = mul nsw i32 %conv18, %conv15
  %add21 = add nsw i32 %mul19, %add30221234
  %add22 = or i32 %hop.0236, 3
  %arrayidx23 = getelementptr inbounds i16, i16* %in, i32 %add22
  %6 = load i16, i16* %arrayidx23, align 2
  %conv24 = sext i16 %6 to i32
  %arrayidx26 = getelementptr inbounds i16, i16* %consts, i32 %add22
  %7 = load i16, i16* %arrayidx26, align 2
  %conv27 = sext i16 %7 to i32
  %mul28 = mul nsw i32 %conv27, %conv24
  %add30 = add nsw i32 %mul28, %add21
  %add31 = or i32 %hop.0236, 4
  %arrayidx32 = getelementptr inbounds i16, i16* %in, i32 %add31
  %8 = load i16, i16* %arrayidx32, align 2
  %conv33 = sext i16 %8 to i32
  %arrayidx35 = getelementptr inbounds i16, i16* %consts, i32 %add31
  %9 = load i16, i16* %arrayidx35, align 2
  %conv36 = sext i16 %9 to i32
  %mul37 = mul nsw i32 %conv36, %conv33
  %add39 = add nsw i32 %mul37, %add48222233
  %add40 = or i32 %hop.0236, 5
  %arrayidx41 = getelementptr inbounds i16, i16* %in, i32 %add40
  %10 = load i16, i16* %arrayidx41, align 2
  %conv42 = sext i16 %10 to i32
  %arrayidx44 = getelementptr inbounds i16, i16* %consts, i32 %add40
  %11 = load i16, i16* %arrayidx44, align 2
  %conv45 = sext i16 %11 to i32
  %mul46 = mul nsw i32 %conv45, %conv42
  %add48 = add nsw i32 %mul46, %add39
  %add49 = or i32 %hop.0236, 6
  %arrayidx50 = getelementptr inbounds i16, i16* %in, i32 %add49
  %12 = load i16, i16* %arrayidx50, align 2
  %conv51 = sext i16 %12 to i32
  %arrayidx53 = getelementptr inbounds i16, i16* %consts, i32 %add49
  %13 = load i16, i16* %arrayidx53, align 2
  %conv54 = sext i16 %13 to i32
  %mul55 = mul nsw i32 %conv54, %conv51
  %add57 = add nsw i32 %mul55, %add66223232
  %add58 = or i32 %hop.0236, 7
  %arrayidx59 = getelementptr inbounds i16, i16* %in, i32 %add58
  %14 = load i16, i16* %arrayidx59, align 2
  %conv60 = sext i16 %14 to i32
  %arrayidx62 = getelementptr inbounds i16, i16* %consts, i32 %add58
  %15 = load i16, i16* %arrayidx62, align 2
  %conv63 = sext i16 %15 to i32
  %mul64 = mul nsw i32 %conv63, %conv60
  %add66 = add nsw i32 %mul64, %add57
  %add67 = or i32 %hop.0236, 8
  %arrayidx68 = getelementptr inbounds i16, i16* %in, i32 %add67
  %16 = load i16, i16* %arrayidx68, align 2
  %conv69 = sext i16 %16 to i32
  %arrayidx71 = getelementptr inbounds i16, i16* %consts, i32 %add67
  %17 = load i16, i16* %arrayidx71, align 2
  %conv72 = sext i16 %17 to i32
  %mul73 = mul nsw i32 %conv72, %conv69
  %add75 = add nsw i32 %mul73, %add84224231
  %add76 = or i32 %hop.0236, 9
  %arrayidx77 = getelementptr inbounds i16, i16* %in, i32 %add76
  %18 = load i16, i16* %arrayidx77, align 2
  %conv78 = sext i16 %18 to i32
  %arrayidx80 = getelementptr inbounds i16, i16* %consts, i32 %add76
  %19 = load i16, i16* %arrayidx80, align 2
  %conv81 = sext i16 %19 to i32
  %mul82 = mul nsw i32 %conv81, %conv78
  %add84 = add nsw i32 %mul82, %add75
  %add85 = or i32 %hop.0236, 10
  %arrayidx86 = getelementptr inbounds i16, i16* %in, i32 %add85
  %20 = load i16, i16* %arrayidx86, align 2
  %conv87 = sext i16 %20 to i32
  %arrayidx89 = getelementptr inbounds i16, i16* %consts, i32 %add85
  %21 = load i16, i16* %arrayidx89, align 2
  %conv90 = sext i16 %21 to i32
  %mul91 = mul nsw i32 %conv90, %conv87
  %add93 = add nsw i32 %mul91, %add102225230
  %add94 = or i32 %hop.0236, 11
  %arrayidx95 = getelementptr inbounds i16, i16* %in, i32 %add94
  %22 = load i16, i16* %arrayidx95, align 2
  %conv96 = sext i16 %22 to i32
  %arrayidx98 = getelementptr inbounds i16, i16* %consts, i32 %add94
  %23 = load i16, i16* %arrayidx98, align 2
  %conv99 = sext i16 %23 to i32
  %mul100 = mul nsw i32 %conv99, %conv96
  %add102 = add nsw i32 %mul100, %add93
  %add103 = or i32 %hop.0236, 12
  %arrayidx104 = getelementptr inbounds i16, i16* %in, i32 %add103
  %24 = load i16, i16* %arrayidx104, align 2
  %conv105 = sext i16 %24 to i32
  %arrayidx107 = getelementptr inbounds i16, i16* %consts, i32 %add103
  %25 = load i16, i16* %arrayidx107, align 2
  %conv108 = sext i16 %25 to i32
  %mul109 = mul nsw i32 %conv108, %conv105
  %add111 = add nsw i32 %mul109, %add120226229
  %add112 = or i32 %hop.0236, 13
  %arrayidx113 = getelementptr inbounds i16, i16* %in, i32 %add112
  %26 = load i16, i16* %arrayidx113, align 2
  %conv114 = sext i16 %26 to i32
  %arrayidx116 = getelementptr inbounds i16, i16* %consts, i32 %add112
  %27 = load i16, i16* %arrayidx116, align 2
  %conv117 = sext i16 %27 to i32
  %mul118 = mul nsw i32 %conv117, %conv114
  %add120 = add nsw i32 %mul118, %add111
  %add121 = or i32 %hop.0236, 14
  %arrayidx122 = getelementptr inbounds i16, i16* %in, i32 %add121
  %28 = load i16, i16* %arrayidx122, align 2
  %conv123 = sext i16 %28 to i32
  %arrayidx125 = getelementptr inbounds i16, i16* %consts, i32 %add121
  %29 = load i16, i16* %arrayidx125, align 2
  %conv126 = sext i16 %29 to i32
  %mul127 = mul nsw i32 %conv126, %conv123
  %add129 = add nsw i32 %mul127, %add138227228
  %add130 = or i32 %hop.0236, 15
  %arrayidx131 = getelementptr inbounds i16, i16* %in, i32 %add130
  %30 = load i16, i16* %arrayidx131, align 2
  %conv132 = sext i16 %30 to i32
  %arrayidx134 = getelementptr inbounds i16, i16* %consts, i32 %add130
  %31 = load i16, i16* %arrayidx134, align 2
  %conv135 = sext i16 %31 to i32
  %mul136 = mul nsw i32 %conv135, %conv132
  %add138 = add nsw i32 %mul136, %add129
  %add139 = add nuw nsw i32 %hop.0236, 16
  %cmp = icmp ult i32 %hop.0236, 64
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Trip count of 8 - does get vectorized
; CHECK-LABEL: tripcount8
; CHECK: LV: Selecting VF: 4
define void @tripcount8(i16* nocapture readonly %in, i32* nocapture %out, i16* nocapture readonly %consts, i32 %n) #0 {
entry:
  %arrayidx20 = getelementptr inbounds i32, i32* %out, i32 1
  %arrayidx38 = getelementptr inbounds i32, i32* %out, i32 2
  %arrayidx56 = getelementptr inbounds i32, i32* %out, i32 3
  %arrayidx74 = getelementptr inbounds i32, i32* %out, i32 4
  %arrayidx92 = getelementptr inbounds i32, i32* %out, i32 5
  %arrayidx110 = getelementptr inbounds i32, i32* %out, i32 6
  %arrayidx128 = getelementptr inbounds i32, i32* %out, i32 7
  %out.promoted = load i32, i32* %out, align 4
  %arrayidx20.promoted = load i32, i32* %arrayidx20, align 4
  %arrayidx38.promoted = load i32, i32* %arrayidx38, align 4
  %arrayidx56.promoted = load i32, i32* %arrayidx56, align 4
  %arrayidx74.promoted = load i32, i32* %arrayidx74, align 4
  %arrayidx92.promoted = load i32, i32* %arrayidx92, align 4
  %arrayidx110.promoted = load i32, i32* %arrayidx110, align 4
  %arrayidx128.promoted = load i32, i32* %arrayidx128, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store i32 %add12, i32* %out, align 4
  store i32 %add30, i32* %arrayidx20, align 4
  store i32 %add48, i32* %arrayidx38, align 4
  store i32 %add66, i32* %arrayidx56, align 4
  store i32 %add84, i32* %arrayidx74, align 4
  store i32 %add102, i32* %arrayidx92, align 4
  store i32 %add120, i32* %arrayidx110, align 4
  store i32 %add138, i32* %arrayidx128, align 4
  ret void

for.body:                                         ; preds = %entry, %for.body
  %hop.0236 = phi i32 [ 0, %entry ], [ %add139, %for.body ]
  %add12220235 = phi i32 [ %out.promoted, %entry ], [ %add12, %for.body ]
  %add30221234 = phi i32 [ %arrayidx20.promoted, %entry ], [ %add30, %for.body ]
  %add48222233 = phi i32 [ %arrayidx38.promoted, %entry ], [ %add48, %for.body ]
  %add66223232 = phi i32 [ %arrayidx56.promoted, %entry ], [ %add66, %for.body ]
  %add84224231 = phi i32 [ %arrayidx74.promoted, %entry ], [ %add84, %for.body ]
  %add102225230 = phi i32 [ %arrayidx92.promoted, %entry ], [ %add102, %for.body ]
  %add120226229 = phi i32 [ %arrayidx110.promoted, %entry ], [ %add120, %for.body ]
  %add138227228 = phi i32 [ %arrayidx128.promoted, %entry ], [ %add138, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %in, i32 %hop.0236
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, i16* %consts, i32 %hop.0236
  %1 = load i16, i16* %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %add12220235
  %add4 = or i32 %hop.0236, 1
  %arrayidx5 = getelementptr inbounds i16, i16* %in, i32 %add4
  %2 = load i16, i16* %arrayidx5, align 2
  %conv6 = sext i16 %2 to i32
  %arrayidx8 = getelementptr inbounds i16, i16* %consts, i32 %add4
  %3 = load i16, i16* %arrayidx8, align 2
  %conv9 = sext i16 %3 to i32
  %mul10 = mul nsw i32 %conv9, %conv6
  %add12 = add nsw i32 %mul10, %add
  %add13 = or i32 %hop.0236, 2
  %arrayidx14 = getelementptr inbounds i16, i16* %in, i32 %add13
  %4 = load i16, i16* %arrayidx14, align 2
  %conv15 = sext i16 %4 to i32
  %arrayidx17 = getelementptr inbounds i16, i16* %consts, i32 %add13
  %5 = load i16, i16* %arrayidx17, align 2
  %conv18 = sext i16 %5 to i32
  %mul19 = mul nsw i32 %conv18, %conv15
  %add21 = add nsw i32 %mul19, %add30221234
  %add22 = or i32 %hop.0236, 3
  %arrayidx23 = getelementptr inbounds i16, i16* %in, i32 %add22
  %6 = load i16, i16* %arrayidx23, align 2
  %conv24 = sext i16 %6 to i32
  %arrayidx26 = getelementptr inbounds i16, i16* %consts, i32 %add22
  %7 = load i16, i16* %arrayidx26, align 2
  %conv27 = sext i16 %7 to i32
  %mul28 = mul nsw i32 %conv27, %conv24
  %add30 = add nsw i32 %mul28, %add21
  %add31 = or i32 %hop.0236, 4
  %arrayidx32 = getelementptr inbounds i16, i16* %in, i32 %add31
  %8 = load i16, i16* %arrayidx32, align 2
  %conv33 = sext i16 %8 to i32
  %arrayidx35 = getelementptr inbounds i16, i16* %consts, i32 %add31
  %9 = load i16, i16* %arrayidx35, align 2
  %conv36 = sext i16 %9 to i32
  %mul37 = mul nsw i32 %conv36, %conv33
  %add39 = add nsw i32 %mul37, %add48222233
  %add40 = or i32 %hop.0236, 5
  %arrayidx41 = getelementptr inbounds i16, i16* %in, i32 %add40
  %10 = load i16, i16* %arrayidx41, align 2
  %conv42 = sext i16 %10 to i32
  %arrayidx44 = getelementptr inbounds i16, i16* %consts, i32 %add40
  %11 = load i16, i16* %arrayidx44, align 2
  %conv45 = sext i16 %11 to i32
  %mul46 = mul nsw i32 %conv45, %conv42
  %add48 = add nsw i32 %mul46, %add39
  %add49 = or i32 %hop.0236, 6
  %arrayidx50 = getelementptr inbounds i16, i16* %in, i32 %add49
  %12 = load i16, i16* %arrayidx50, align 2
  %conv51 = sext i16 %12 to i32
  %arrayidx53 = getelementptr inbounds i16, i16* %consts, i32 %add49
  %13 = load i16, i16* %arrayidx53, align 2
  %conv54 = sext i16 %13 to i32
  %mul55 = mul nsw i32 %conv54, %conv51
  %add57 = add nsw i32 %mul55, %add66223232
  %add58 = or i32 %hop.0236, 7
  %arrayidx59 = getelementptr inbounds i16, i16* %in, i32 %add58
  %14 = load i16, i16* %arrayidx59, align 2
  %conv60 = sext i16 %14 to i32
  %arrayidx62 = getelementptr inbounds i16, i16* %consts, i32 %add58
  %15 = load i16, i16* %arrayidx62, align 2
  %conv63 = sext i16 %15 to i32
  %mul64 = mul nsw i32 %conv63, %conv60
  %add66 = add nsw i32 %mul64, %add57
  %add67 = or i32 %hop.0236, 8
  %arrayidx68 = getelementptr inbounds i16, i16* %in, i32 %add67
  %16 = load i16, i16* %arrayidx68, align 2
  %conv69 = sext i16 %16 to i32
  %arrayidx71 = getelementptr inbounds i16, i16* %consts, i32 %add67
  %17 = load i16, i16* %arrayidx71, align 2
  %conv72 = sext i16 %17 to i32
  %mul73 = mul nsw i32 %conv72, %conv69
  %add75 = add nsw i32 %mul73, %add84224231
  %add76 = or i32 %hop.0236, 9
  %arrayidx77 = getelementptr inbounds i16, i16* %in, i32 %add76
  %18 = load i16, i16* %arrayidx77, align 2
  %conv78 = sext i16 %18 to i32
  %arrayidx80 = getelementptr inbounds i16, i16* %consts, i32 %add76
  %19 = load i16, i16* %arrayidx80, align 2
  %conv81 = sext i16 %19 to i32
  %mul82 = mul nsw i32 %conv81, %conv78
  %add84 = add nsw i32 %mul82, %add75
  %add85 = or i32 %hop.0236, 10
  %arrayidx86 = getelementptr inbounds i16, i16* %in, i32 %add85
  %20 = load i16, i16* %arrayidx86, align 2
  %conv87 = sext i16 %20 to i32
  %arrayidx89 = getelementptr inbounds i16, i16* %consts, i32 %add85
  %21 = load i16, i16* %arrayidx89, align 2
  %conv90 = sext i16 %21 to i32
  %mul91 = mul nsw i32 %conv90, %conv87
  %add93 = add nsw i32 %mul91, %add102225230
  %add94 = or i32 %hop.0236, 11
  %arrayidx95 = getelementptr inbounds i16, i16* %in, i32 %add94
  %22 = load i16, i16* %arrayidx95, align 2
  %conv96 = sext i16 %22 to i32
  %arrayidx98 = getelementptr inbounds i16, i16* %consts, i32 %add94
  %23 = load i16, i16* %arrayidx98, align 2
  %conv99 = sext i16 %23 to i32
  %mul100 = mul nsw i32 %conv99, %conv96
  %add102 = add nsw i32 %mul100, %add93
  %add103 = or i32 %hop.0236, 12
  %arrayidx104 = getelementptr inbounds i16, i16* %in, i32 %add103
  %24 = load i16, i16* %arrayidx104, align 2
  %conv105 = sext i16 %24 to i32
  %arrayidx107 = getelementptr inbounds i16, i16* %consts, i32 %add103
  %25 = load i16, i16* %arrayidx107, align 2
  %conv108 = sext i16 %25 to i32
  %mul109 = mul nsw i32 %conv108, %conv105
  %add111 = add nsw i32 %mul109, %add120226229
  %add112 = or i32 %hop.0236, 13
  %arrayidx113 = getelementptr inbounds i16, i16* %in, i32 %add112
  %26 = load i16, i16* %arrayidx113, align 2
  %conv114 = sext i16 %26 to i32
  %arrayidx116 = getelementptr inbounds i16, i16* %consts, i32 %add112
  %27 = load i16, i16* %arrayidx116, align 2
  %conv117 = sext i16 %27 to i32
  %mul118 = mul nsw i32 %conv117, %conv114
  %add120 = add nsw i32 %mul118, %add111
  %add121 = or i32 %hop.0236, 14
  %arrayidx122 = getelementptr inbounds i16, i16* %in, i32 %add121
  %28 = load i16, i16* %arrayidx122, align 2
  %conv123 = sext i16 %28 to i32
  %arrayidx125 = getelementptr inbounds i16, i16* %consts, i32 %add121
  %29 = load i16, i16* %arrayidx125, align 2
  %conv126 = sext i16 %29 to i32
  %mul127 = mul nsw i32 %conv126, %conv123
  %add129 = add nsw i32 %mul127, %add138227228
  %add130 = or i32 %hop.0236, 15
  %arrayidx131 = getelementptr inbounds i16, i16* %in, i32 %add130
  %30 = load i16, i16* %arrayidx131, align 2
  %conv132 = sext i16 %30 to i32
  %arrayidx134 = getelementptr inbounds i16, i16* %consts, i32 %add130
  %31 = load i16, i16* %arrayidx134, align 2
  %conv135 = sext i16 %31 to i32
  %mul136 = mul nsw i32 %conv135, %conv132
  %add138 = add nsw i32 %mul136, %add129
  %add139 = add nuw nsw i32 %hop.0236, 16
  %cmp = icmp ult i32 %hop.0236, 112
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

attributes #0 = { "target-features"="+mve" }