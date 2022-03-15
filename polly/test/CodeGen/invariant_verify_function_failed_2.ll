; RUN: opt %loadPolly -S -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s -check-prefix=SCOPS
; RUN: opt %loadPolly -S -polly-codegen -polly-invariant-load-hoisting=true %s | FileCheck %s
;
; Check we generate valid code.

; SCOPS:         Statements {
; SCOPS-NEXT:     	Stmt_if_then2457
; SCOPS-NEXT:             Domain :=
; SCOPS-NEXT:                 [p_0] -> { Stmt_if_then2457[] : p_0 = 1 };
; SCOPS-NEXT:             Schedule :=
; SCOPS-NEXT:                 [p_0] -> { Stmt_if_then2457[] -> [1] };
; SCOPS-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; SCOPS-NEXT:                 [p_0] -> { Stmt_if_then2457[] -> MemRef_sub2464[] };
; SCOPS-NEXT:     	Stmt_cond_false2468
; SCOPS-NEXT:             Domain :=
; SCOPS-NEXT:                 [p_0] -> { Stmt_cond_false2468[] : p_0 = 1 };
; SCOPS-NEXT:             Schedule :=
; SCOPS-NEXT:                 [p_0] -> { Stmt_cond_false2468[] -> [2] };
; SCOPS-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; SCOPS-NEXT:                 [p_0] -> { Stmt_cond_false2468[] -> MemRef_sub2464[] };
; SCOPS-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOPS-NEXT:                 [p_0] -> { Stmt_cond_false2468[] -> MemRef_A[0] };
; SCOPS-NEXT:     	Stmt_if_else2493
; SCOPS-NEXT:             Domain :=
; SCOPS-NEXT:                 [p_0] -> { Stmt_if_else2493[] : p_0 >= 2 or p_0 = 0 };
; SCOPS-NEXT:             Schedule :=
; SCOPS-NEXT:                 [p_0] -> { Stmt_if_else2493[] -> [0] : p_0 >= 2 or p_0 = 0 };
; SCOPS-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOPS-NEXT:                 [p_0] -> { Stmt_if_else2493[] -> MemRef_B[0] };
; SCOPS-NEXT:     }

; CHECK: polly.start

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { i32, i32, i32, i32, i32, i32, [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16**, i16*, i16*, i16**, i16**, i16***, i8*, i16***, i64***, i64***, i16****, i8**, i8**, %struct.s*, %struct.s*, %struct.s*, i32, i32, i32, i32, i32, i32, i32 }

@enc_picture = external global %struct.s*, align 8

; Function Attrs: nounwind uwtable
define void @compute_colocated(%struct.s*** %listX, i1* %A, i32* %B) #0 {
entry:
  br label %for.body2414

for.body2414:                                     ; preds = %for.inc2621, %entry
  %indvars.iv902 = phi i64 [ %indvars.iv.next903, %for.inc2621 ], [ 0, %entry ]
  br label %if.else2454

if.else2454:                                      ; preds = %for.body2414
  %cmp2455 = icmp eq i64 %indvars.iv902, 2
  br i1 %cmp2455, label %if.then2457, label %if.else2493

if.then2457:                                      ; preds = %if.else2454
  %arrayidx2461 = getelementptr inbounds %struct.s**, %struct.s*** %listX, i64 %indvars.iv902
  %tmp1 = load %struct.s**, %struct.s*** %arrayidx2461, align 8, !tbaa !1
  %arrayidx2462 = getelementptr inbounds %struct.s*, %struct.s** %tmp1, i64 0
  %tmp2 = load %struct.s*, %struct.s** %arrayidx2462, align 8, !tbaa !1
  %poc2463 = getelementptr inbounds %struct.s, %struct.s* %tmp2, i64 0, i32 1
  %tmp3 = load i32, i32* %poc2463, align 4, !tbaa !5
  %sub2464 = sub nsw i32 0, %tmp3
  br label %cond.false2468

cond.false2468:                                   ; preds = %if.then2457
  %cmp2477 = icmp sgt i32 %sub2464, 127
  store i1 %cmp2477, i1* %A
  br label %for.inc2621

if.else2493:                                      ; preds = %if.else2454
  %arrayidx2497 = getelementptr inbounds %struct.s**, %struct.s*** %listX, i64 %indvars.iv902
  %tmp4 = load %struct.s**, %struct.s*** %arrayidx2497, align 8, !tbaa !1
  %arrayidx2498 = getelementptr inbounds %struct.s*, %struct.s** %tmp4, i64 0
  %tmp5 = load %struct.s*, %struct.s** %arrayidx2498, align 8, !tbaa !1
  %poc2499 = getelementptr inbounds %struct.s, %struct.s* %tmp5, i64 0, i32 1
  %tmp6 = load i32, i32* %poc2499, align 4, !tbaa !5
  store i32 %tmp6, i32* %B
  br label %for.inc2621

for.inc2621:                                      ; preds = %if.else2493, %cond.false2468
  %indvars.iv.next903 = add nuw nsw i64 %indvars.iv902, 2
  br i1 undef, label %for.body2414, label %if.end2624

if.end2624:                                       ; preds = %for.inc2621
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !7, i64 4}
!6 = !{!"storable_picture", !3, i64 0, !7, i64 4, !7, i64 8, !7, i64 12, !7, i64 16, !7, i64 20, !3, i64 24, !3, i64 1608, !3, i64 3192, !3, i64 4776, !7, i64 6360, !7, i64 6364, !7, i64 6368, !7, i64 6372, !7, i64 6376, !7, i64 6380, !7, i64 6384, !7, i64 6388, !7, i64 6392, !7, i64 6396, !7, i64 6400, !7, i64 6404, !7, i64 6408, !7, i64 6412, !7, i64 6416, !2, i64 6424, !2, i64 6432, !2, i64 6440, !2, i64 6448, !2, i64 6456, !2, i64 6464, !2, i64 6472, !2, i64 6480, !2, i64 6488, !2, i64 6496, !2, i64 6504, !2, i64 6512, !2, i64 6520, !2, i64 6528, !2, i64 6536, !2, i64 6544, !7, i64 6552, !7, i64 6556, !7, i64 6560, !7, i64 6564, !7, i64 6568, !7, i64 6572, !7, i64 6576}
!7 = !{!"int", !3, i64 0}
