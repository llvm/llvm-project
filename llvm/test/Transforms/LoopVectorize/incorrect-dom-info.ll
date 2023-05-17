; This test is based on one of benchmarks from SPEC2006. It exposes a bug with
; incorrect updating of the dom-tree.
; RUN: opt < %s -passes=loop-vectorize -verify-dom-info
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@PL_utf8skip = external constant [0 x i8]

; Function Attrs: nounwind ssp uwtable
define void @Perl_pp_quotemeta() #0 {
  %len = alloca i64, align 8
  br i1 undef, label %2, label %1

; <label>:1                                       ; preds = %0
  br label %3

; <label>:2                                       ; preds = %0
  br label %3

; <label>:3                                       ; preds = %2, %1
  br i1 undef, label %34, label %4

; <label>:4                                       ; preds = %3
  br i1 undef, label %5, label %6

; <label>:5                                       ; preds = %4
  br label %6

; <label>:6                                       ; preds = %5, %4
  br i1 undef, label %7, label %8

; <label>:7                                       ; preds = %6
  br label %8

; <label>:8                                       ; preds = %7, %6
  br i1 undef, label %.preheader, label %9

.preheader:                                       ; preds = %9, %8
  br i1 undef, label %.loopexit, label %.lr.ph

; <label>:9                                       ; preds = %8
  br i1 undef, label %thread-pre-split.preheader, label %.preheader

thread-pre-split.preheader:                       ; preds = %9
  br i1 undef, label %thread-pre-split._crit_edge, label %.lr.ph21

.thread-pre-split.loopexit_crit_edge:             ; preds = %19
  %scevgep.sum = xor i64 %umax, -1
  %scevgep45 = getelementptr i8, ptr %d.020, i64 %scevgep.sum
  br label %thread-pre-split.loopexit

thread-pre-split.loopexit:                        ; preds = %11, %.thread-pre-split.loopexit_crit_edge
  %d.1.lcssa = phi ptr [ %scevgep45, %.thread-pre-split.loopexit_crit_edge ], [ %d.020, %11 ]
  br i1 false, label %thread-pre-split._crit_edge, label %.lr.ph21

.lr.ph21:                                         ; preds = %26, %thread-pre-split.loopexit, %thread-pre-split.preheader
  %d.020 = phi ptr [ undef, %26 ], [ %d.1.lcssa, %thread-pre-split.loopexit ], [ undef, %thread-pre-split.preheader ]
  %10 = phi i64 [ %28, %26 ], [ undef, %thread-pre-split.loopexit ], [ undef, %thread-pre-split.preheader ]
  br i1 undef, label %11, label %22

; <label>:11                                      ; preds = %.lr.ph21
  %12 = getelementptr inbounds [0 x i8], ptr @PL_utf8skip, i64 0, i64 undef
  %13 = load i8, ptr %12, align 1
  %14 = zext i8 %13 to i64
  %15 = icmp ugt i64 %14, %10
  %. = select i1 %15, i64 %10, i64 %14
  br i1 undef, label %thread-pre-split.loopexit, label %.lr.ph28

.lr.ph28:                                         ; preds = %11
  %16 = xor i64 %10, -1
  %17 = xor i64 %14, -1
  %18 = icmp ugt i64 %16, %17
  %umax = select i1 %18, i64 %16, i64 %17
  br label %19

; <label>:19                                      ; preds = %19, %.lr.ph28
  %ulen.126 = phi i64 [ %., %.lr.ph28 ], [ %20, %19 ]
  %20 = add i64 %ulen.126, -1
  %21 = icmp eq i64 %20, 0
  br i1 %21, label %.thread-pre-split.loopexit_crit_edge, label %19

; <label>:22                                      ; preds = %.lr.ph21
  br i1 undef, label %26, label %23

; <label>:23                                      ; preds = %22
  br i1 undef, label %26, label %24

; <label>:24                                      ; preds = %23
  br i1 undef, label %26, label %25

; <label>:25                                      ; preds = %24
  br label %26

; <label>:26                                      ; preds = %25, %24, %23, %22
  %27 = load i64, ptr %len, align 8
  %28 = add i64 %27, -1
  br i1 undef, label %thread-pre-split._crit_edge, label %.lr.ph21

thread-pre-split._crit_edge:                      ; preds = %26, %thread-pre-split.loopexit, %thread-pre-split.preheader
  br label %.loopexit

.lr.ph:                                           ; preds = %33, %.preheader
  br i1 undef, label %29, label %thread-pre-split5

; <label>:29                                      ; preds = %.lr.ph
  br i1 undef, label %33, label %30

; <label>:30                                      ; preds = %29
  br i1 undef, label %33, label %31

thread-pre-split5:                                ; preds = %.lr.ph
  br i1 undef, label %33, label %31

; <label>:31                                      ; preds = %thread-pre-split5, %30
  br i1 undef, label %33, label %32

; <label>:32                                      ; preds = %31
  br label %33

; <label>:33                                      ; preds = %32, %31, %thread-pre-split5, %30, %29
  br i1 undef, label %.loopexit, label %.lr.ph

.loopexit:                                        ; preds = %33, %thread-pre-split._crit_edge, %.preheader
  br label %35

; <label>:34                                      ; preds = %3
  br label %35

; <label>:35                                      ; preds = %34, %.loopexit
  br i1 undef, label %37, label %36

; <label>:36                                      ; preds = %35
  br label %37

; <label>:37                                      ; preds = %36, %35
  ret void
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.6.0 "}
