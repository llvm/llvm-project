; RUN: opt < %s -S -passes='print<scalar-evolution>,loop-unroll<O3>' 2>/dev/null | FileCheck %s --check-prefix=CHECK-LCSSA

; Test that forgetLcssaPhiWithNewPredecessor correctly handles LCSSA PHIs with
; non-SCEV-able struct types from with.overflow intrinsics. Previously the
; function bailed early on struct-typed PHIs without invalidating their
; extractvalue users, causing a stale SCEV cache entry and assertion failure
; in getAddRecExpr.
define void @lcssa_phi_uadd_with_overflow(ptr %p) {
; CHECK-LCSSA-LABEL: define void @lcssa_phi_uadd_with_overflow
entry:
  br label %lbl_entry

lbl_entry:
  %0 = phi i64 [ 0, %lbl_entry ], [ 1, %entry ]
  %1 = load i64, ptr %p, align 8
  %2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 0, i64 %1)
  br i1 true, label %lbl_entry, label %if.end

if.end:
  %3 = extractvalue { i64, i1 } %2, 0
  br label %lbl_br5.peel

lbl_br5.peel:
  %4 = phi i8 [ 1, %lbl_br5.peel ], [ 0, %if.end ]
  %ov11.1.peel = phi i64 [ %sub.peel, %lbl_br5.peel ], [ %3, %if.end ]
  %sub.peel = add i64 %ov11.1.peel, 1
  br label %lbl_br5.peel
}

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
