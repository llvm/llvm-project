; RUN: opt -S -licm -licm-coldness-threshold=0 < %s | FileCheck %s --check-prefix=CHECK-LICM
; RUN: opt -S -licm < %s | opt -S -loop-sink | FileCheck %s --check-prefix=CHECK-SINK
; RUN: opt -S < %s -passes='require<opt-remark-emit>,loop-mssa(licm),loop-sink' \
; RUN:     | FileCheck %s --check-prefix=CHECK-SINK
; RUN: opt -S -licm -licm-coldness-threshold=0 -verify-memoryssa < %s | FileCheck %s --check-prefix=CHECK-LICM
; RUN: opt -S -licm -verify-memoryssa < %s | FileCheck %s --check-prefix=CHECK-BFI-LICM

; Original source code:
; int g;
; int foo(int p, int x) {
;   for (int i = 0; i != x; i++)
;     if (__builtin_expect(i == p, 0)) {
;       x += g; x *= g;
;     }
;   return x;
; }
;
; Load of global value g should not be hoisted to preheader.

@g = global i32 0, align 4

define i32 @foo(i32, i32) #0 !prof !2 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %._crit_edge, label %.lr.ph.preheader

.lr.ph.preheader:
  br label %.lr.ph

; CHECK-LICM: .lr.ph.preheader:
; CHECK-LICM: load i32, i32* @g
; CHECK-LICM: br label %.lr.ph

; CHECK-BFI-LICM: .lr.ph.preheader:
; CHECK-BFI-LICM-NOT: load i32, i32* @g
; CHECK-BFI-LICM: br label %.lr.ph

.lr.ph:
  %.03 = phi i32 [ %8, %.combine ], [ 0, %.lr.ph.preheader ]
  %.012 = phi i32 [ %.1, %.combine ], [ %1, %.lr.ph.preheader ]
  %4 = icmp eq i32 %.03, %0
  br i1 %4, label %.then, label %.combine, !prof !1

.then:
  %5 = load i32, i32* @g, align 4
  %6 = add nsw i32 %5, %.012
  %7 = mul nsw i32 %6, %5
  br label %.combine

; CHECK-SINK: .then:
; CHECK-SINK: load i32, i32* @g
; CHECK-SINK: br label %.combine

.combine:
  %.1 = phi i32 [ %7, %.then ], [ %.012, %.lr.ph ]
  %8 = add nuw nsw i32 %.03, 1
  %9 = icmp eq i32 %8, %.1
  br i1 %9, label %._crit_edge.loopexit, label %.lr.ph

._crit_edge.loopexit:
  %.1.lcssa = phi i32 [ %.1, %.combine ]
  br label %._crit_edge

._crit_edge:
  %.01.lcssa = phi i32 [ 0, %2 ], [ %.1.lcssa, %._crit_edge.loopexit ]
  ret i32 %.01.lcssa
}

!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"function_entry_count", i64 1}
