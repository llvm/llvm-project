; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@c = external unnamed_addr global i32, align 4
@b = external global [1 x i32], align 4

; Function Attrs: nounwind
define void @fn2() #0 align 4 {
  br i1 undef, label %.lr.ph, label %4

; We used to crash because a bad DAGCombine was creating i32-typed SETCC nodes,
; even when crbits are enabled.
; CHECK-LABEL: @fn2
; CHECK: blr

.lr.ph:                                           ; preds = %0
  br i1 undef, label %.lr.ph.split, label %.preheader

.preheader:                                       ; preds = %.preheader, %.lr.ph
  br i1 undef, label %.lr.ph.split, label %.preheader

.lr.ph.split:                                     ; preds = %.preheader, %.lr.ph
  br i1 undef, label %._crit_edge, label %.lr.ph.split.split

.lr.ph.split.split:                               ; preds = %.lr.ph.split.split, %.lr.ph.split
  %1 = phi i32 [ %2, %.lr.ph.split.split ], [ undef, %.lr.ph.split ]
  %cmp = icmp eq ptr @c, @b
  %constexpr = select i1 %cmp, i1 true, i1 false
  %constexpr1 = zext i1 %constexpr to i32
  %constexpr2 = and i32 %constexpr1, %constexpr1
  %constexpr3 = and i32 %constexpr2, %constexpr1
  %constexpr4 = and i32 %constexpr3, %constexpr1
  %constexpr5 = and i32 %constexpr4, %constexpr1
  %constexpr6 = and i32 %constexpr5, %constexpr1
  %constexpr7 = and i32 %constexpr6, %constexpr1
  %constexpr8 = and i32 %constexpr7, %constexpr1
  %2 = and i32 %1, %constexpr8
  %3 = icmp slt i32 undef, 4
  br i1 %3, label %.lr.ph.split.split, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph.split.split, %.lr.ph.split
  %.lcssa = phi i32 [ undef, %.lr.ph.split ], [ %2, %.lr.ph.split.split ]
  br label %4

4:                                                ; preds = %._crit_edge, %0
  ret void
}

attributes #0 = { nounwind "target-cpu"="ppc64le" }

