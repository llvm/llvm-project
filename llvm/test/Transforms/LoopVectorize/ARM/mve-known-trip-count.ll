; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; Trip count of 5 - shouldn't be vectorized.
; CHECK-LABEL: tripcount5
; CHECK: LV: Selecting VF: 1
define void @tripcount5(ptr nocapture readonly %in, ptr nocapture %out, ptr nocapture readonly %consts, i32 %n) #0 {
entry:
  %arrayidx20 = getelementptr inbounds i32, ptr %out, i32 1
  %arrayidx38 = getelementptr inbounds i32, ptr %out, i32 2
  %arrayidx56 = getelementptr inbounds i32, ptr %out, i32 3
  %arrayidx74 = getelementptr inbounds i32, ptr %out, i32 4
  %arrayidx92 = getelementptr inbounds i32, ptr %out, i32 5
  %arrayidx110 = getelementptr inbounds i32, ptr %out, i32 6
  %arrayidx128 = getelementptr inbounds i32, ptr %out, i32 7
  %out.promoted = load i32, ptr %out, align 4
  %arrayidx20.promoted = load i32, ptr %arrayidx20, align 4
  %arrayidx38.promoted = load i32, ptr %arrayidx38, align 4
  %arrayidx56.promoted = load i32, ptr %arrayidx56, align 4
  %arrayidx74.promoted = load i32, ptr %arrayidx74, align 4
  %arrayidx92.promoted = load i32, ptr %arrayidx92, align 4
  %arrayidx110.promoted = load i32, ptr %arrayidx110, align 4
  %arrayidx128.promoted = load i32, ptr %arrayidx128, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store i32 %add12, ptr %out, align 4
  store i32 %add30, ptr %arrayidx20, align 4
  store i32 %add48, ptr %arrayidx38, align 4
  store i32 %add66, ptr %arrayidx56, align 4
  store i32 %add84, ptr %arrayidx74, align 4
  store i32 %add102, ptr %arrayidx92, align 4
  store i32 %add120, ptr %arrayidx110, align 4
  store i32 %add138, ptr %arrayidx128, align 4
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
  %arrayidx = getelementptr inbounds i16, ptr %in, i32 %hop.0236
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, ptr %consts, i32 %hop.0236
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %add12220235
  %add4 = or i32 %hop.0236, 1
  %arrayidx5 = getelementptr inbounds i16, ptr %in, i32 %add4
  %2 = load i16, ptr %arrayidx5, align 2
  %conv6 = sext i16 %2 to i32
  %arrayidx8 = getelementptr inbounds i16, ptr %consts, i32 %add4
  %3 = load i16, ptr %arrayidx8, align 2
  %conv9 = sext i16 %3 to i32
  %mul10 = mul nsw i32 %conv9, %conv6
  %add12 = add nsw i32 %mul10, %add
  %add13 = or i32 %hop.0236, 2
  %arrayidx14 = getelementptr inbounds i16, ptr %in, i32 %add13
  %4 = load i16, ptr %arrayidx14, align 2
  %conv15 = sext i16 %4 to i32
  %arrayidx17 = getelementptr inbounds i16, ptr %consts, i32 %add13
  %5 = load i16, ptr %arrayidx17, align 2
  %conv18 = sext i16 %5 to i32
  %mul19 = mul nsw i32 %conv18, %conv15
  %add21 = add nsw i32 %mul19, %add30221234
  %add22 = or i32 %hop.0236, 3
  %arrayidx23 = getelementptr inbounds i16, ptr %in, i32 %add22
  %6 = load i16, ptr %arrayidx23, align 2
  %conv24 = sext i16 %6 to i32
  %arrayidx26 = getelementptr inbounds i16, ptr %consts, i32 %add22
  %7 = load i16, ptr %arrayidx26, align 2
  %conv27 = sext i16 %7 to i32
  %mul28 = mul nsw i32 %conv27, %conv24
  %add30 = add nsw i32 %mul28, %add21
  %add31 = or i32 %hop.0236, 4
  %arrayidx32 = getelementptr inbounds i16, ptr %in, i32 %add31
  %8 = load i16, ptr %arrayidx32, align 2
  %conv33 = sext i16 %8 to i32
  %arrayidx35 = getelementptr inbounds i16, ptr %consts, i32 %add31
  %9 = load i16, ptr %arrayidx35, align 2
  %conv36 = sext i16 %9 to i32
  %mul37 = mul nsw i32 %conv36, %conv33
  %add39 = add nsw i32 %mul37, %add48222233
  %add40 = or i32 %hop.0236, 5
  %arrayidx41 = getelementptr inbounds i16, ptr %in, i32 %add40
  %10 = load i16, ptr %arrayidx41, align 2
  %conv42 = sext i16 %10 to i32
  %arrayidx44 = getelementptr inbounds i16, ptr %consts, i32 %add40
  %11 = load i16, ptr %arrayidx44, align 2
  %conv45 = sext i16 %11 to i32
  %mul46 = mul nsw i32 %conv45, %conv42
  %add48 = add nsw i32 %mul46, %add39
  %add49 = or i32 %hop.0236, 6
  %arrayidx50 = getelementptr inbounds i16, ptr %in, i32 %add49
  %12 = load i16, ptr %arrayidx50, align 2
  %conv51 = sext i16 %12 to i32
  %arrayidx53 = getelementptr inbounds i16, ptr %consts, i32 %add49
  %13 = load i16, ptr %arrayidx53, align 2
  %conv54 = sext i16 %13 to i32
  %mul55 = mul nsw i32 %conv54, %conv51
  %add57 = add nsw i32 %mul55, %add66223232
  %add58 = or i32 %hop.0236, 7
  %arrayidx59 = getelementptr inbounds i16, ptr %in, i32 %add58
  %14 = load i16, ptr %arrayidx59, align 2
  %conv60 = sext i16 %14 to i32
  %arrayidx62 = getelementptr inbounds i16, ptr %consts, i32 %add58
  %15 = load i16, ptr %arrayidx62, align 2
  %conv63 = sext i16 %15 to i32
  %mul64 = mul nsw i32 %conv63, %conv60
  %add66 = add nsw i32 %mul64, %add57
  %add67 = or i32 %hop.0236, 8
  %arrayidx68 = getelementptr inbounds i16, ptr %in, i32 %add67
  %16 = load i16, ptr %arrayidx68, align 2
  %conv69 = sext i16 %16 to i32
  %arrayidx71 = getelementptr inbounds i16, ptr %consts, i32 %add67
  %17 = load i16, ptr %arrayidx71, align 2
  %conv72 = sext i16 %17 to i32
  %mul73 = mul nsw i32 %conv72, %conv69
  %add75 = add nsw i32 %mul73, %add84224231
  %add76 = or i32 %hop.0236, 9
  %arrayidx77 = getelementptr inbounds i16, ptr %in, i32 %add76
  %18 = load i16, ptr %arrayidx77, align 2
  %conv78 = sext i16 %18 to i32
  %arrayidx80 = getelementptr inbounds i16, ptr %consts, i32 %add76
  %19 = load i16, ptr %arrayidx80, align 2
  %conv81 = sext i16 %19 to i32
  %mul82 = mul nsw i32 %conv81, %conv78
  %add84 = add nsw i32 %mul82, %add75
  %add85 = or i32 %hop.0236, 10
  %arrayidx86 = getelementptr inbounds i16, ptr %in, i32 %add85
  %20 = load i16, ptr %arrayidx86, align 2
  %conv87 = sext i16 %20 to i32
  %arrayidx89 = getelementptr inbounds i16, ptr %consts, i32 %add85
  %21 = load i16, ptr %arrayidx89, align 2
  %conv90 = sext i16 %21 to i32
  %mul91 = mul nsw i32 %conv90, %conv87
  %add93 = add nsw i32 %mul91, %add102225230
  %add94 = or i32 %hop.0236, 11
  %arrayidx95 = getelementptr inbounds i16, ptr %in, i32 %add94
  %22 = load i16, ptr %arrayidx95, align 2
  %conv96 = sext i16 %22 to i32
  %arrayidx98 = getelementptr inbounds i16, ptr %consts, i32 %add94
  %23 = load i16, ptr %arrayidx98, align 2
  %conv99 = sext i16 %23 to i32
  %mul100 = mul nsw i32 %conv99, %conv96
  %add102 = add nsw i32 %mul100, %add93
  %add103 = or i32 %hop.0236, 12
  %arrayidx104 = getelementptr inbounds i16, ptr %in, i32 %add103
  %24 = load i16, ptr %arrayidx104, align 2
  %conv105 = sext i16 %24 to i32
  %arrayidx107 = getelementptr inbounds i16, ptr %consts, i32 %add103
  %25 = load i16, ptr %arrayidx107, align 2
  %conv108 = sext i16 %25 to i32
  %mul109 = mul nsw i32 %conv108, %conv105
  %add111 = add nsw i32 %mul109, %add120226229
  %add112 = or i32 %hop.0236, 13
  %arrayidx113 = getelementptr inbounds i16, ptr %in, i32 %add112
  %26 = load i16, ptr %arrayidx113, align 2
  %conv114 = sext i16 %26 to i32
  %arrayidx116 = getelementptr inbounds i16, ptr %consts, i32 %add112
  %27 = load i16, ptr %arrayidx116, align 2
  %conv117 = sext i16 %27 to i32
  %mul118 = mul nsw i32 %conv117, %conv114
  %add120 = add nsw i32 %mul118, %add111
  %add121 = or i32 %hop.0236, 14
  %arrayidx122 = getelementptr inbounds i16, ptr %in, i32 %add121
  %28 = load i16, ptr %arrayidx122, align 2
  %conv123 = sext i16 %28 to i32
  %arrayidx125 = getelementptr inbounds i16, ptr %consts, i32 %add121
  %29 = load i16, ptr %arrayidx125, align 2
  %conv126 = sext i16 %29 to i32
  %mul127 = mul nsw i32 %conv126, %conv123
  %add129 = add nsw i32 %mul127, %add138227228
  %add130 = or i32 %hop.0236, 15
  %arrayidx131 = getelementptr inbounds i16, ptr %in, i32 %add130
  %30 = load i16, ptr %arrayidx131, align 2
  %conv132 = sext i16 %30 to i32
  %arrayidx134 = getelementptr inbounds i16, ptr %consts, i32 %add130
  %31 = load i16, ptr %arrayidx134, align 2
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
define void @tripcount8(ptr nocapture readonly %in, ptr nocapture %out, ptr nocapture readonly %consts, i32 %n) #0 {
entry:
  %out.promoted = load i32, ptr %out, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store i32 %add12, ptr %out, align 4
  ret void

for.body:                                         ; preds = %entry, %for.body
  %hop.0236 = phi i32 [ 0, %entry ], [ %add139, %for.body ]
  %add12220235 = phi i32 [ %out.promoted, %entry ], [ %add12, %for.body ]
  %arrayidx = getelementptr inbounds i16, ptr %in, i32 %hop.0236
  %0 = load i16, ptr %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds i16, ptr %consts, i32 %hop.0236
  %1 = load i16, ptr %arrayidx1, align 2
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %add12220235
  %add4 = or i32 %hop.0236, 1
  %arrayidx5 = getelementptr inbounds i16, ptr %in, i32 %add4
  %2 = load i16, ptr %arrayidx5, align 2
  %conv6 = sext i16 %2 to i32
  %arrayidx8 = getelementptr inbounds i16, ptr %consts, i32 %add4
  %3 = load i16, ptr %arrayidx8, align 2
  %conv9 = sext i16 %3 to i32
  %mul10 = mul nsw i32 %conv9, %conv6
  %add12 = add nsw i32 %mul10, %add
  %add139 = add nuw nsw i32 %hop.0236, 16
  %cmp = icmp ult i32 %hop.0236, 112
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Larger example with predication that should also not be vectorized
; CHECK-LABEL: predicated_test
; CHECK: LV: Selecting VF: 1
; CHECK: LV: Selecting VF: 1
define dso_local i32 @predicated_test(i32 noundef %0, ptr %glob) #0 {
  %2 = alloca [101 x i32], align 4
  %3 = alloca [21 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %159

5:                                                ; preds = %1
  %6 = getelementptr inbounds [21 x i32], ptr %3, i32 0, i32 5
  br label %7

7:                                                ; preds = %5, %155
  %8 = phi i32 [ %10, %155 ], [ undef, %5 ]
  %9 = phi i32 [ %156, %155 ], [ 0, %5 ]
  %10 = shl i32 %8, 4
  store i32 %10, ptr %6, align 4
  br label %11

11:                                               ; preds = %7, %152
  %12 = phi i32 [ 0, %7 ], [ %153, %152 ]
  %13 = lshr i32 %12, 3
  %14 = getelementptr inbounds [21 x i32], ptr %3, i32 0, i32 %13
  %15 = load i32, ptr %14, align 4
  %16 = shl nuw nsw i32 %12, 2
  %17 = and i32 %16, 28
  %18 = ashr i32 %15, %17
  %19 = and i32 %18, 15
  %20 = mul nuw nsw i32 %19, 5
  %21 = add nuw nsw i32 %20, 5
  %22 = getelementptr inbounds i32, ptr %glob, i32 %21
  %23 = mul nuw nsw i32 %12, 5
  br label %24

24:                                               ; preds = %11, %78
  %25 = phi i32 [ 0, %11 ], [ %79, %78 ]
  %26 = add nuw nsw i32 %25, %23
  %27 = getelementptr inbounds [101 x i32], ptr %2, i32 0, i32 %26
  store i32 0, ptr %27, align 4
  %28 = getelementptr inbounds i32, ptr %22, i32 %25
  %29 = load i32, ptr %28, align 4
  %30 = and i32 %29, 1
  %31 = icmp eq i32 %30, 0
  %32 = and i32 %29, 2
  %33 = icmp eq i32 %32, 0
  %34 = and i32 %29, 4
  %35 = icmp eq i32 %34, 0
  %36 = and i32 %29, 8
  %37 = icmp eq i32 %36, 0
  %38 = and i32 %29, 16
  %39 = icmp eq i32 %38, 0
  %40 = and i32 %29, 32
  %41 = icmp eq i32 %40, 0
  %42 = and i32 %29, 64
  %43 = icmp eq i32 %42, 0
  %44 = and i32 %29, 128
  %45 = icmp eq i32 %44, 0
  %46 = and i32 %29, 254
  %47 = icmp eq i32 %46, 0
  br i1 %31, label %48, label %62

48:                                               ; preds = %24
  %49 = select i1 %33, i32 0, i32 129
  %50 = or i32 %49, 258
  %51 = select i1 %35, i32 %49, i32 %50
  %52 = or i32 %51, 516
  %53 = select i1 %37, i32 %51, i32 %52
  %54 = or i32 %53, 1032
  %55 = select i1 %39, i32 %53, i32 %54
  %56 = or i32 %55, 2064
  %57 = select i1 %41, i32 %55, i32 %56
  %58 = or i32 %57, 4128
  %59 = select i1 %43, i32 %57, i32 %58
  %60 = or i32 %59, 8256
  %61 = select i1 %45, i32 %59, i32 %60
  br i1 %47, label %78, label %76

62:                                               ; preds = %24
  %63 = select i1 %33, i32 0, i32 516
  %64 = or i32 %63, 1032
  %65 = select i1 %35, i32 %63, i32 %64
  %66 = or i32 %65, 2064
  %67 = select i1 %37, i32 %65, i32 %66
  %68 = or i32 %67, 4128
  %69 = select i1 %39, i32 %67, i32 %68
  %70 = or i32 %69, 8256
  %71 = select i1 %41, i32 %69, i32 %70
  %72 = or i32 %71, 16512
  %73 = select i1 %43, i32 %71, i32 %72
  %74 = or i32 %73, 33024
  %75 = select i1 %45, i32 %73, i32 %74
  br i1 %47, label %78, label %76

76:                                               ; preds = %62, %48
  %77 = phi i32 [ %61, %48 ], [ %75, %62 ]
  store i32 %77, ptr %27, align 4
  br label %78

78:                                               ; preds = %76, %62, %48
  %79 = add nuw nsw i32 %25, 1
  %80 = icmp eq i32 %79, 5
  br i1 %80, label %81, label %24

81:                                               ; preds = %78
  br label %82

82:                                               ; preds = %81, %149
  %83 = phi i32 [ %150, %149 ], [ 0, %81 ]
  %84 = add nuw nsw i32 %83, %23
  %85 = getelementptr inbounds [101 x i32], ptr %2, i32 0, i32 %84
  %86 = load i32, ptr %85, align 4
  %87 = shl i32 %86, 30
  %88 = and i32 %87, 1073741824
  %89 = and i32 %86, 2
  %90 = icmp eq i32 %89, 0
  %91 = select i1 %90, i32 %88, i32 1073741824
  %92 = shl i32 %86, 27
  %93 = and i32 %92, 536870912
  %94 = or i32 %91, %93
  %95 = shl i32 %86, 25
  %96 = and i32 %95, 268435456
  %97 = or i32 %94, %96
  %98 = shl i32 %86, 23
  %99 = and i32 %98, 134217728
  %100 = or i32 %97, %99
  %101 = or i32 %100, %86
  %102 = and i32 %86, 31
  %103 = and i32 %101, 32
  %104 = shl nuw nsw i32 %103, 21
  %105 = or i32 %102, %103
  %106 = and i32 %101, 64
  %107 = shl nuw nsw i32 %106, 19
  %108 = or i32 %104, %107
  %109 = or i32 %105, %106
  %110 = and i32 %101, 128
  %111 = shl nuw nsw i32 %110, 17
  %112 = or i32 %108, %111
  %113 = or i32 %109, %110
  %114 = and i32 %101, 256
  %115 = shl nuw nsw i32 %114, 15
  %116 = or i32 %112, %115
  %117 = or i32 %113, %114
  %118 = and i32 %101, 512
  %119 = shl nuw nsw i32 %118, 13
  %120 = or i32 %116, %119
  %121 = or i32 %120, %101
  %122 = or i32 %117, %118
  %123 = and i32 %121, 1024
  %124 = shl nuw nsw i32 %123, 11
  %125 = or i32 %122, %123
  %126 = and i32 %121, 2048
  %127 = shl nuw nsw i32 %126, 9
  %128 = or i32 %124, %127
  %129 = or i32 %125, %126
  %130 = and i32 %121, 4096
  %131 = shl nuw nsw i32 %130, 7
  %132 = or i32 %128, %131
  %133 = or i32 %129, %130
  %134 = and i32 %121, 8192
  %135 = shl nuw nsw i32 %134, 5
  %136 = or i32 %132, %135
  %137 = or i32 %136, %121
  %138 = or i32 %133, %134
  %139 = and i32 %137, 16384
  %140 = or i32 %138, %139
  %141 = and i32 %137, 32768
  %142 = or i32 %140, %141
  %143 = icmp eq i32 %142, 0
  br i1 %143, label %149, label %144

144:                                              ; preds = %82
  %145 = shl nuw nsw i32 %139, 3
  %146 = shl nuw nsw i32 %141, 1
  %147 = or i32 %145, %146
  %148 = or i32 %147, %137
  store i32 %148, ptr %85, align 4
  br label %149

149:                                              ; preds = %82, %144
  %150 = add nuw nsw i32 %83, 1
  %151 = icmp eq i32 %150, 5
  br i1 %151, label %152, label %82

152:                                              ; preds = %149
  %153 = add nuw nsw i32 %12, 1
  %154 = icmp eq i32 %153, 20
  br i1 %154, label %155, label %11

155:                                              ; preds = %152
  %156 = add nuw nsw i32 %9, 1
  %157 = icmp eq i32 %156, %0
  br i1 %157, label %158, label %7

158:                                              ; preds = %155
  br label %159

159:                                              ; preds = %158, %1
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  ret i32 0
}

; This has a maximum trip count of 4. The codegen is currently much better with <8 x half> vectorization.
; CHECK-LABEL: arm_q15_to_f16_remainder
; CHECK: LV: Selecting VF: 8
define void @arm_q15_to_f16_remainder(ptr nocapture noundef readonly %pSrc, ptr nocapture noundef writeonly noalias %pDst, i32 noundef %blockSize) #0 {
entry:
  %rem = and i32 %blockSize, 3
  %cmp.not5 = icmp eq i32 %rem, 0
  br i1 %cmp.not5, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %blkCnt.08 = phi i32 [ %dec, %while.body ], [ %rem, %while.body.preheader ]
  %pIn.07 = phi ptr [ %incdec.ptr, %while.body ], [ %pSrc, %while.body.preheader ]
  %pDst.addr.06 = phi ptr [ %incdec.ptr2, %while.body ], [ %pDst, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, ptr %pIn.07, i32 2
  %0 = load i16, ptr %pIn.07, align 2
  %conv1 = sitofp i16 %0 to half
  %1 = fmul fast half %conv1, 0xH0200
  %incdec.ptr2 = getelementptr inbounds i8, ptr %pDst.addr.06, i32 2
  store half %1, ptr %pDst.addr.06, align 2
  %dec = add nsw i32 %blkCnt.08, -1
  %cmp.not = icmp eq i32 %dec, 0
  br i1 %cmp.not, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}


declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)

attributes #0 = { "target-features"="+mve.fp" }
