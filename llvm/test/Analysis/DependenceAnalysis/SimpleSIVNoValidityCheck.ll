; RUN: opt < %s -disable-output -passes="print<da>"                            \
; RUN: -da-disable-delinearization-checks 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -passes="print<da>"                            \
; RUN: 2>&1 | FileCheck --check-prefix=LIN %s

; CHECK-LABEL: t1
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [1 -2]!
; CHECK: da analyze - none!

; LIN-LABEL: t1
; LIN: da analyze - input [* *]!
; LIN: da analyze - anti [* *|<]!
; LIN: da analyze - output [* *]!

;; void t1(int n, int m, int a[][m]) {
;;   for (int i = 0; i < n-1; ++i)
;;     for (int j = 2; j < m; ++j)
;;       a[i][j] = a[i+1][j-2];
;; }

define void @t1(i32 signext %n, i32 signext %m, i32* %a) {
entry:
  %0 = zext i32 %m to i64
  %1 = sext i32 %m to i64
  %sub = add nsw i32 %n, -1
  %2 = sext i32 %sub to i64
  %cmp7 = icmp slt i64 0, %2
  br i1 %cmp7, label %for.body, label %for.end14

for.body:                                         ; preds = %entry, %for.inc12
  %indvars.iv28 = phi i64 [ %indvars.iv.next3, %for.inc12 ], [ 0, %entry ]
  %cmp25 = icmp slt i64 2, %1
  br i1 %cmp25, label %for.body4, label %for.inc12

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv6 = phi i64 [ %indvars.iv.next, %for.body4 ], [ 2, %for.body ]
  %3 = add nuw nsw i64 %indvars.iv28, 1
  %4 = mul nuw nsw i64 %3, %0
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %4
  %5 = add nsw i64 %indvars.iv6, -2
  %arrayidx7 = getelementptr inbounds i32, i32* %arrayidx, i64 %5
  %6 = load i32, i32* %arrayidx7, align 4
  %7 = mul nuw nsw i64 %indvars.iv28, %0
  %arrayidx9 = getelementptr inbounds i32, i32* %a, i64 %7
  %arrayidx11 = getelementptr inbounds i32, i32* %arrayidx9, i64 %indvars.iv6
  store i32 %6, i32* %arrayidx11, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv6, 1
  %cmp2 = icmp slt i64 %indvars.iv.next, %1
  br i1 %cmp2, label %for.body4, label %for.inc12

for.inc12:                                        ; preds = %for.body4, %for.body
  %indvars.iv29 = phi i64 [ %indvars.iv28, %for.body ], [ %indvars.iv28, %for.body4 ]
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv29, 1
  %cmp = icmp slt i64 %indvars.iv.next3, %2
  br i1 %cmp, label %for.body, label %for.end14

for.end14:                                        ; preds = %entry, %for.inc12
  ret void
}

; CHECK-LABEL: t2
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [1 -2 0 -3 2]!
; CHECK: da analyze - none!

; LIN-LABEL: t2
; LIN: da analyze - input [* * * * *]!
; LIN: da analyze - anti [* * * * *|<]!
; LIN: da analyze - output [* * * * *]!

;; void t2(int n, int m, int a[][n][n][n][m]) {
;;   for (int i1 = 0; i1 < n-1; ++i1)
;;     for (int i2 = 2; i2 < n; ++i2)
;;       for (int i3 = 0; i3 < n; ++i3)
;;         for (int i4 = 3; i4 < n; ++i4)
;;           for (int i5 = 0; i5 < m-2; ++i5)
;;             a[i1][i2][i3][i4][i5] = a[i1+1][i2-2][i3][i4-3][i5+2];
;; }

define void @t2(i32 signext %n, i32 signext %m, i32* %a) {
entry:
  %0 = zext i32 %n to i64
  %1 = zext i32 %n to i64
  %2 = zext i32 %n to i64
  %3 = zext i32 %m to i64
  %4 = sext i32 %n to i64
  %sub = add nsw i32 %n, -1
  %5 = sext i32 %sub to i64
  %cmp26 = icmp slt i64 0, %5
  br i1 %cmp26, label %for.body, label %for.end50

for.body:                                         ; preds = %entry, %for.inc48
  %indvars.iv1227 = phi i64 [ %indvars.iv.next13, %for.inc48 ], [ 0, %entry ]
  %cmp223 = icmp slt i64 2, %4
  br i1 %cmp223, label %for.body4, label %for.inc48

for.body4:                                        ; preds = %for.body, %for.inc45
  %indvars.iv924 = phi i64 [ %indvars.iv.next10, %for.inc45 ], [ 2, %for.body ]
  %wide.trip.count7 = zext i32 %n to i64
  %exitcond820 = icmp ne i64 0, %wide.trip.count7
  br i1 %exitcond820, label %for.body8, label %for.inc45

for.body8:                                        ; preds = %for.body4, %for.inc42
  %indvars.iv521 = phi i64 [ %indvars.iv.next6, %for.inc42 ], [ 0, %for.body4 ]
  %wide.trip.count = zext i32 %n to i64
  %exitcond17 = icmp ne i64 3, %wide.trip.count
  br i1 %exitcond17, label %for.body12, label %for.inc42

for.body12:                                       ; preds = %for.body8, %for.inc39
  %indvars.iv218 = phi i64 [ %indvars.iv.next3, %for.inc39 ], [ 3, %for.body8 ]
  %sub14 = add nsw i32 %m, -2
  %6 = sext i32 %sub14 to i64
  %cmp1515 = icmp slt i64 0, %6
  br i1 %cmp1515, label %for.body17, label %for.inc39

for.body17:                                       ; preds = %for.body12, %for.body17
  %indvars.iv16 = phi i64 [ %indvars.iv.next, %for.body17 ], [ 0, %for.body12 ]
  %7 = add nuw nsw i64 %indvars.iv1227, 1
  %8 = mul nuw i64 %0, %1
  %9 = mul nuw i64 %8, %2
  %10 = mul nuw i64 %9, %3
  %11 = mul nsw i64 %10, %7
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %11
  %12 = add nsw i64 %indvars.iv924, -2
  %13 = mul nuw i64 %1, %2
  %14 = mul nuw i64 %13, %3
  %15 = mul nsw i64 %14, %12
  %arrayidx20 = getelementptr inbounds i32, i32* %arrayidx, i64 %15
  %16 = mul nuw i64 %2, %3
  %17 = mul nsw i64 %16, %indvars.iv521
  %arrayidx22 = getelementptr inbounds i32, i32* %arrayidx20, i64 %17
  %18 = add nsw i64 %indvars.iv218, -3
  %19 = mul nuw nsw i64 %18, %3
  %arrayidx25 = getelementptr inbounds i32, i32* %arrayidx22, i64 %19
  %20 = add nuw nsw i64 %indvars.iv16, 2
  %arrayidx28 = getelementptr inbounds i32, i32* %arrayidx25, i64 %20
  %21 = load i32, i32* %arrayidx28, align 4
  %22 = mul nuw i64 %0, %1
  %23 = mul nuw i64 %22, %2
  %24 = mul nuw i64 %23, %3
  %25 = mul nsw i64 %24, %indvars.iv1227
  %arrayidx30 = getelementptr inbounds i32, i32* %a, i64 %25
  %26 = mul nuw i64 %1, %2
  %27 = mul nuw i64 %26, %3
  %28 = mul nsw i64 %27, %indvars.iv924
  %arrayidx32 = getelementptr inbounds i32, i32* %arrayidx30, i64 %28
  %29 = mul nuw i64 %2, %3
  %30 = mul nsw i64 %29, %indvars.iv521
  %arrayidx34 = getelementptr inbounds i32, i32* %arrayidx32, i64 %30
  %31 = mul nuw nsw i64 %indvars.iv218, %3
  %arrayidx36 = getelementptr inbounds i32, i32* %arrayidx34, i64 %31
  %arrayidx38 = getelementptr inbounds i32, i32* %arrayidx36, i64 %indvars.iv16
  store i32 %21, i32* %arrayidx38, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv16, 1
  %cmp15 = icmp slt i64 %indvars.iv.next, %6
  br i1 %cmp15, label %for.body17, label %for.inc39

for.inc39:                                        ; preds = %for.body17, %for.body12
  %indvars.iv219 = phi i64 [ %indvars.iv218, %for.body12 ], [ %indvars.iv218, %for.body17 ]
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv219, 1
  %exitcond = icmp ne i64 %indvars.iv.next3, %wide.trip.count
  br i1 %exitcond, label %for.body12, label %for.inc42

for.inc42:                                        ; preds = %for.inc39, %for.body8
  %indvars.iv522 = phi i64 [ %indvars.iv521, %for.body8 ], [ %indvars.iv521, %for.inc39 ]
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv522, 1
  %exitcond8 = icmp ne i64 %indvars.iv.next6, %wide.trip.count7
  br i1 %exitcond8, label %for.body8, label %for.inc45

for.inc45:                                        ; preds = %for.inc42, %for.body4
  %indvars.iv925 = phi i64 [ %indvars.iv924, %for.body4 ], [ %indvars.iv924, %for.inc42 ]
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv925, 1
  %cmp2 = icmp slt i64 %indvars.iv.next10, %4
  br i1 %cmp2, label %for.body4, label %for.inc48

for.inc48:                                        ; preds = %for.inc45, %for.body
  %indvars.iv1228 = phi i64 [ %indvars.iv1227, %for.body ], [ %indvars.iv1227, %for.inc45 ]
  %indvars.iv.next13 = add nuw nsw i64 %indvars.iv1228, 1
  %cmp = icmp slt i64 %indvars.iv.next13, %5
  br i1 %cmp, label %for.body, label %for.end50

for.end50:                                        ; preds = %entry, %for.inc48
  ret void
}


; CHECK-LABEL: t3
; CHECK: da analyze - none!
; CHECK: da analyze - consistent anti [1 -2]!
; CHECK: da analyze - none!

; LIN-LABEL: t3
; LIN: da analyze - input [* *]!
; LIN: da analyze - anti [* *|<]!
; LIN: da analyze - output [* *]!

;; // No sign or zero extension, but with compile-time unknown loop lower bound.
;; void t3(unsigned long long n, unsigned long long m, unsigned long long lb, float a[][m]) {
;;   for (unsigned long long i = 0; i < n-1; ++i)
;;     for (unsigned long long j = lb; j < m; ++j)
;;       a[i][j] = a[i+1][j-2];
;; }

define void @t3(i64 %n, i64 %m, i64 %lb, float* %a) {
entry:
  %0 = add i64 %n, -1
  %exitcond3 = icmp ne i64 0, %0
  br i1 %exitcond3, label %for.body.preheader, label %for.end11

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc9
  %i.04 = phi i64 [ %inc10, %for.inc9 ], [ 0, %for.body.preheader ]
  %cmp21 = icmp ult i64 %lb, %m
  br i1 %cmp21, label %for.body4.preheader, label %for.inc9

for.body4.preheader:                              ; preds = %for.body
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j.02 = phi i64 [ %inc, %for.body4 ], [ %lb, %for.body4.preheader ]
  %add = add i64 %i.04, 1
  %1 = mul nsw i64 %add, %m
  %arrayidx = getelementptr inbounds float, float* %a, i64 %1
  %sub5 = add i64 %j.02, -2
  %arrayidx6 = getelementptr inbounds float, float* %arrayidx, i64 %sub5
  %2 = bitcast float* %arrayidx6 to i32*
  %3 = load i32, i32* %2, align 4
  %4 = mul nsw i64 %i.04, %m
  %arrayidx7 = getelementptr inbounds float, float* %a, i64 %4
  %arrayidx8 = getelementptr inbounds float, float* %arrayidx7, i64 %j.02
  %5 = bitcast float* %arrayidx8 to i32*
  store i32 %3, i32* %5, align 4
  %inc = add i64 %j.02, 1
  %cmp2 = icmp ult i64 %inc, %m
  br i1 %cmp2, label %for.body4, label %for.inc9.loopexit

for.inc9.loopexit:                                ; preds = %for.body4
  br label %for.inc9

for.inc9:                                         ; preds = %for.inc9.loopexit, %for.body
  %inc10 = add i64 %i.04, 1
  %exitcond = icmp ne i64 %inc10, %0
  br i1 %exitcond, label %for.body, label %for.end11.loopexit

for.end11.loopexit:                               ; preds = %for.inc9
  br label %for.end11

for.end11:                                        ; preds = %for.end11.loopexit, %entry
  ret void
}

