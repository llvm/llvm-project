; RUN: opt %loadPolly -polly-print-instructions -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:    Statements {
; CHECK-NEXT:   Stmt_Region__TO__Stmt
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_Region__TO__Stmt[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_Region__TO__Stmt[i0] -> [i0, 0] };
; CHECK-NEXT:       MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:           { Stmt_Region__TO__Stmt[i0] -> MemRef_C[0] };
; CHECK-NEXT:       Instructions {
; CHECK-NEXT:             %cond = fcmp oeq double 2.100000e+01, 2.100000e+01
; CHECK-NEXT:       }
; CHECK-NEXT:   Stmt_Stmt
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_Stmt[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_Stmt[i0] -> [i0, 1] };
; CHECK-NEXT:       MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:           { Stmt_Stmt[i0] -> MemRef_A[i0] };
; CHECK-NEXT:       Instructions {
; CHECK-NEXT:             store i32 %i.0, ptr %arrayidx, align 4, !polly_split_after !0
; CHECK-NEXT:       }
; CHECK-NEXT:   Stmt_Stmt_b
; CHECK-NEXT:       Domain :=
; CHECK-NEXT:           { Stmt_Stmt_b[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:       Schedule :=
; CHECK-NEXT:           { Stmt_Stmt_b[i0] -> [i0, 2] };
; CHECK-NEXT:       MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:           { Stmt_Stmt_b[i0] -> MemRef_B[i0] };
; CHECK-NEXT:       Instructions {
; CHECK-NEXT:             store i32 %i.0, ptr %arrayidx2, align 4
; CHECK-NEXT:       }
; CHECK-NEXT:   }
;
; Function Attrs: noinline nounwind uwtable
define void @func(ptr %A, ptr %B, ptr %C) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %Region

Region:                                           ; preds = %for.body
  %cond = fcmp oeq double 2.100000e+01, 2.100000e+01
  br i1 %cond, label %Region_true, label %Stmt

Region_true:                                      ; preds = %Region
  store double 0.000000e+00, ptr %C
  br label %Stmt

Stmt:                                             ; preds = %Region_true, %Region
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  store i32 %i.0, ptr %arrayidx, align 4, !polly_split_after !0
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %idxprom1
  store i32 %i.0, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %Stmt
  %add = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

!0 = !{!"polly_split_after"}
