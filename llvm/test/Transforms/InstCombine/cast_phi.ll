; RUN: opt < %s -instcombine -S | FileCheck %s
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define void @MainKernel(i32 %iNumSteps, i32 %tid, i32 %base) {
; CHECK-NOT: bitcast

  %callA = alloca [258 x float], align 4
  %callB = alloca [258 x float], align 4
  %conv.i = uitofp i32 %iNumSteps to float
  %1 = bitcast float %conv.i to i32
  %conv.i12 = zext i32 %tid to i64
  %arrayidx3 = getelementptr inbounds [258 x float], [258 x float]* %callA, i64 0, i64 %conv.i12
  %2 = bitcast float* %arrayidx3 to i32*
  store i32 %1, i32* %2, align 4
  %arrayidx6 = getelementptr inbounds [258 x float], [258 x float]* %callB, i64 0, i64 %conv.i12
  %3 = bitcast float* %arrayidx6 to i32*
  store i32 %1, i32* %3, align 4
  %cmp7 = icmp eq i32 %tid, 0
  br i1 %cmp7, label %.bb1, label %.bb2

.bb1:
  %arrayidx10 = getelementptr inbounds [258 x float], [258 x float]* %callA, i64 0, i64 256
  store float %conv.i, float* %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds [258 x float], [258 x float]* %callB, i64 0, i64 256
  store float 0.000000e+00, float* %arrayidx11, align 4
  br label %.bb2

.bb2:
  %cmp135 = icmp sgt i32 %iNumSteps, 0
  br i1 %cmp135, label %.bb3, label %.bb8

; CHECK-LABEL: .bb3
; CHECK: phi float
; CHECK: phi float
; CHECK: phi i32 {{.*}} [ %iNumSteps
; CHECK-NOT: rA.sroa.[0-9].[0-9] = phi i32
; CHECK-NOT: phi float
; CHECK-NOT: phi i32
; CHECK-LABEL: .bb4

.bb3:
  %rA.sroa.8.0 = phi i32 [ %rA.sroa.8.2, %.bb12 ], [ %1, %.bb2 ]
  %rA.sroa.0.0 = phi i32 [ %rA.sroa.0.2, %.bb12 ], [ %1, %.bb2 ]
  %i12.06 = phi i32 [ %sub, %.bb12 ], [ %iNumSteps, %.bb2 ]
  %4 = icmp ugt i32 %i12.06, %base
  %add = add i32 %i12.06, 1
  %conv.i9 = sext i32 %add to i64
  %arrayidx20 = getelementptr inbounds [258 x float], [258 x float]* %callA, i64 0, i64 %conv.i9
  %5 = bitcast float* %arrayidx20 to i32*
  %arrayidx24 = getelementptr inbounds [258 x float], [258 x float]* %callB, i64 0, i64 %conv.i9
  %6 = bitcast float* %arrayidx24 to i32*
  %cmp40 = icmp ult i32 %i12.06, %base
  br i1 %4, label %.bb4, label %.bb5

.bb4:
  %7 = load i32, i32* %5, align 4
  %8 = load i32, i32* %6, align 4
  %9 = bitcast i32 %8 to float
  %10 = bitcast i32 %7 to float
  %add33 = fadd float %9, %10
  %11 = bitcast i32 %rA.sroa.8.0 to float
  %add33.1 = fadd float %add33, %11
  %12 = bitcast float %add33.1 to i32
  %13 = bitcast i32 %rA.sroa.0.0 to float
  %add33.2 = fadd float %add33.1, %13
  %14 = bitcast float %add33.2 to i32
  br label %.bb5

; CHECK-LABEL: .bb5
; CHECK: phi float
; CHECK: phi float
; CHECK-NOT: rA.sroa.[0-9].[0-9] = phi i32
; CHECK-NOT: phi float
; CHECK-NOT: phi i32
; CHECK-LABEL: .bb6

.bb5:
  %rA.sroa.8.1 = phi i32 [ %12, %.bb4 ], [ %rA.sroa.8.0, %.bb3 ]
  %rA.sroa.0.1 = phi i32 [ %14, %.bb4 ], [ %rA.sroa.0.0, %.bb3 ]
  br i1 %cmp40, label %.bb6, label %.bb7

.bb6:
  store i32 %rA.sroa.0.1, i32* %2, align 4
  store i32 %rA.sroa.8.1, i32* %3, align 4
  br label %.bb7

.bb7:
  br i1 %4, label %.bb9, label %.bb10

.bb8:
  ret void

.bb9:
  %15 = load i32, i32* %5, align 4
  %16 = load i32, i32* %6, align 4
  %17 = bitcast i32 %16 to float
  %18 = bitcast i32 %15 to float
  %add33.112 = fadd float %17, %18
  %19 = bitcast i32 %rA.sroa.8.1 to float
  %add33.1.1 = fadd float %add33.112, %19
  %20 = bitcast float %add33.1.1 to i32
  %21 = bitcast i32 %rA.sroa.0.1 to float
  %add33.2.1 = fadd float %add33.1.1, %21
  %22 = bitcast float %add33.2.1 to i32
  br label %.bb10

; CHECK-LABEL: .bb10
; CHECK: phi float
; CHECK: phi float
; CHECK-NOT: rA.sroa.[0-9].[0-9] = phi i32
; CHECK-NOT: phi float
; CHECK-NOT: phi i32
; CHECK-LABEL: .bb11

.bb10:
  %rA.sroa.8.2 = phi i32 [ %20, %.bb9 ], [ %rA.sroa.8.1, %.bb7 ]
  %rA.sroa.0.2 = phi i32 [ %22, %.bb9 ], [ %rA.sroa.0.1, %.bb7 ]
  br i1 %cmp40, label %.bb11, label %.bb12

; CHECK-LABEL: .bb11
; CHECK: store float
; CHECK: store float
; CHECK-NOT: store i32 %rA.sroa.[0-9].[0-9]
; CHECK-LABEL: .bb12

.bb11:
  store i32 %rA.sroa.0.2, i32* %2, align 4
  store i32 %rA.sroa.8.2, i32* %3, align 4
  br label %.bb12

.bb12:
  %sub = add i32 %i12.06, -4
  %cmp13 = icmp sgt i32 %sub, 0
  br i1 %cmp13, label %.bb3, label %.bb8
}
