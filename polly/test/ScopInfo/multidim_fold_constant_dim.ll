; RUN: opt %loadPolly -polly-scops -analyze -S < %s | FileCheck %s
;
;    struct com {
;      double Real;
;      double Img;
;    };
;
;    void foo(long n, struct com A[][n]) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < 1000; j++)
;          A[i][j].Real += A[i][j].Img;
;    }
;
;    int main() {
;      struct com A[100][1000];
;      foo(1000, A);
;    }

; CHECK:      Arrays {
; CHECK-NEXT:     double MemRef_A[*][(2 * %n)]; // Element size 8
; CHECK-NEXT: }

; CHECK: 	Stmt_for_body3
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_for_body3[i0, i1] : 0 <= i0 <= 99 and 0 <= i1 <= 999 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_for_body3[i0, i1] -> [i0, i1] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body3[i0, i1] -> MemRef_A[i0, 1 + 2i1] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body3[i0, i1] -> MemRef_A[i0, 2i1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_body3[i0, i1] -> MemRef_A[i0, 2i1] };

source_filename = "/tmp/test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.com = type { double, double }

define void @foo(i64 %n, %struct.com* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc7, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.end9

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %j.0, 1000
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp = mul nsw i64 %i.0, %n
  %arrayidx = getelementptr inbounds %struct.com, %struct.com* %A, i64 %tmp
  %arrayidx4 = getelementptr inbounds %struct.com, %struct.com* %arrayidx, i64 %j.0
  %Img = getelementptr inbounds %struct.com, %struct.com* %arrayidx4, i64 0, i32 1
  %tmp2 = load double, double* %Img, align 8
  %tmp3 = mul nsw i64 %i.0, %n
  %arrayidx5 = getelementptr inbounds %struct.com, %struct.com* %A, i64 %tmp3
  %arrayidx6 = getelementptr inbounds %struct.com, %struct.com* %arrayidx5, i64 %j.0
  %Real = getelementptr inbounds %struct.com, %struct.com* %arrayidx6, i64 0, i32 0
  %tmp4 = load double, double* %Real, align 8
  %add = fadd double %tmp4, %tmp2
  store double %add, double* %Real, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nuw nsw i64 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc7

for.inc7:                                         ; preds = %for.end
  %inc8 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end9:                                         ; preds = %for.cond
  ret void
}

define i32 @main() {
entry:
  %A = alloca [100 x [1000 x %struct.com]], align 16
  %tmp = getelementptr inbounds [100 x [1000 x %struct.com]], [100 x [1000 x %struct.com]]* %A, i64 0, i64 0, i64 0
  call void @foo(i64 1000, %struct.com* nonnull %tmp)
  ret i32 0
}
