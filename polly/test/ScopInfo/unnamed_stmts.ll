; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

; This test case verifies that we generate numbered statement names in case
; no LLVM-IR names are used in the test case. We also verify, that we
; distinguish statements named with a number and unnamed statements that happen
; to have an index identical to a number used in a statement name.

; CHECK: Arrays {
; CHECK-NEXT:     float MemRef0[*][%n]; // Element size 4
; CHECK-NEXT:     float MemRef1[*][%n]; // Element size 4
; CHECK-NEXT: }
; CHECK-NEXT: Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     float MemRef0[*][ [n] -> { [] -> [(n)] } ]; // Element size 4
; CHECK-NEXT:     float MemRef1[*][ [n] -> { [] -> [(n)] } ]; // Element size 4
; CHECK-NEXT: }

; CHECK: Statements {
; CHECK-NEXT: 	Stmt2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt2[i0, i1] : 0 <= i0 < n and 0 <= i1 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt2[i0, i1] -> [0, i0, i1, 0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt2[i0, i1] -> MemRef0[i0, i1] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt2[i0, i1] -> MemRef1[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt2[i0, i1] -> MemRef1[i0, i1] };
; CHECK-NEXT: 	Stmt10
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt10[i0, i1] : 0 <= i0 < n and 0 <= i1 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt10[i0, i1] -> [1, i0, i1, 0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt10[i0, i1] -> MemRef1[i0, i1] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt10[i0, i1] -> MemRef0[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt10[i0, i1] -> MemRef0[i0, i1] };
; CHECK-NEXT: 	Stmt_2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_2[i0, i1] : 0 <= i0 < n and 0 <= i1 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_2[i0, i1] -> [1, i0, i1, 1] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_2[i0, i1] -> MemRef0[i0, i1]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @vec3(i64 %n, ptr, ptr) #0 {
  br label %.split

.split:                                           ; preds = %0
  br label %.preheader2.lr.ph

.preheader2.lr.ph:                                ; preds = %.split
  br label %.preheader2

.preheader2:                                      ; preds = %.preheader2.lr.ph, %15
  %i.010 = phi i64 [ 0, %.preheader2.lr.ph ], [ %16, %15 ]
  br label %.lr.ph8

.lr.ph8:                                          ; preds = %.preheader2
  br label %4

..preheader1_crit_edge:                           ; preds = %15
  br label %.preheader1

.preheader1:                                      ; preds = %..preheader1_crit_edge, %.split
  %3 = icmp sgt i64 %n, 0
  br i1 %3, label %.preheader.lr.ph, label %"name"

.preheader.lr.ph:                                 ; preds = %.preheader1
  br label %.preheader

; <label>:4:                                      ; preds = %.lr.ph8, %4
  %j.07 = phi i64 [ 0, %.lr.ph8 ], [ %14, %4 ]
  %5 = mul nsw i64 %i.010, %n
  %6 = getelementptr inbounds float, ptr %1, i64 %5
  %7 = getelementptr inbounds float, ptr %6, i64 %j.07
  %8 = load float, ptr %7, align 4
  %9 = mul nsw i64 %i.010, %n
  %10 = getelementptr inbounds float, ptr %0, i64 %9
  %11 = getelementptr inbounds float, ptr %10, i64 %j.07
  %12 = load float, ptr %11, align 4
  %13 = fadd float %8, %12
  store float %13, ptr %11, align 4
  %14 = add nuw nsw i64 %j.07, 1
  %exitcond13 = icmp ne i64 %14, %n
  br i1 %exitcond13, label %4, label %._crit_edge9

._crit_edge9:                                     ; preds = %4
  br label %15

; <label>:15:                                     ; preds = %._crit_edge9, %.preheader2
  %16 = add nuw nsw i64 %i.010, 1
  %exitcond14 = icmp ne i64 %16, %n
  br i1 %exitcond14, label %.preheader2, label %..preheader1_crit_edge

.preheader:                                       ; preds = %.preheader.lr.ph, %29
  %i1.04 = phi i64 [ 0, %.preheader.lr.ph ], [ %30, %29 ]
  %17 = icmp sgt i64 %n, 0
  br i1 %17, label %.lr.ph, label %29

.lr.ph:                                           ; preds = %.preheader
  br label %18

; <label>:18:                                     ; preds = %.lr.ph, %18
  %j2.03 = phi i64 [ 0, %.lr.ph ], [ %28, %"2" ]
  %19 = mul nsw i64 %i1.04, %n
  %20 = getelementptr inbounds float, ptr %0, i64 %19
  %21 = getelementptr inbounds float, ptr %20, i64 %j2.03
  %22 = load float, ptr %21, align 4
  %23 = mul nsw i64 %i1.04, %n
  %24 = getelementptr inbounds float, ptr %1, i64 %23
  %25 = getelementptr inbounds float, ptr %24, i64 %j2.03
  %26 = load float, ptr %25, align 4
  %27 = fadd float %22, %26
  store float %27, ptr %25, align 4
  br label %"2"

"2":
  store float 42.0, ptr %25
  %28 = add nuw nsw i64 %j2.03, 1
  %exitcond = icmp ne i64 %28, %n
  br i1 %exitcond, label %18, label %._crit_edge

._crit_edge:                                      ; preds = %18
  br label %29

; <label>:29:                                     ; preds = %._crit_edge, %.preheader
  %30 = add nuw nsw i64 %i1.04, 1
  %exitcond12 = icmp ne i64 %30, %n
  br i1 %exitcond12, label %.preheader, label %._crit_edge6

._crit_edge6:                                     ; preds = %29
  br label %"name"

"name":
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"Ubuntu clang version 3.7.1-3ubuntu4 (tags/RELEASE_371/final) (based on LLVM 3.7.1)"}
