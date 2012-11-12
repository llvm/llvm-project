; RUN: llc < %s -mtriple=arm-apple-darwin -mcpu=cortex-a8 | FileCheck %s
; rdar://8598427
; Adjust if-converter heuristics to avoid predicating vmrs which can cause
; significant regression.

%struct.xyz_t = type { double, double, double }

define i32 @effie(i32 %tsets, %struct.xyz_t* nocapture %p, i32 %a, i32 %b, i32 %c) nounwind readonly noinline {
; CHECK: effie:
entry:
  %0 = icmp sgt i32 %tsets, 0
  br i1 %0, label %bb.nph, label %bb6

bb.nph:                                           ; preds = %entry
  %1 = add nsw i32 %b, %a
  %2 = add nsw i32 %1, %c
  br label %bb

bb:                                               ; preds = %bb4, %bb.nph
; CHECK: vcmpe.f64
; CHECK: vmrs APSR_nzcv, fpscr
  %r.19 = phi i32 [ 0, %bb.nph ], [ %r.0, %bb4 ]
  %n.08 = phi i32 [ 0, %bb.nph ], [ %10, %bb4 ]
  %scevgep10 = getelementptr inbounds %struct.xyz_t* %p, i32 %n.08, i32 0
  %scevgep11 = getelementptr %struct.xyz_t* %p, i32 %n.08, i32 1
  %3 = load double* %scevgep10, align 4
  %4 = load double* %scevgep11, align 4
  %5 = fcmp uge double %3, %4
  br i1 %5, label %bb3, label %bb1

bb1:                                              ; preds = %bb
; CHECK-NOT: it
; CHECK-NOT: vcmpemi
; CHECK-NOT: vmrsmi
; CHECK: vcmpe.f64
; CHECK: vmrs APSR_nzcv, fpscr
  %scevgep12 = getelementptr %struct.xyz_t* %p, i32 %n.08, i32 2
  %6 = load double* %scevgep12, align 4
  %7 = fcmp uge double %3, %6
  br i1 %7, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  %8 = add nsw i32 %2, %r.19
  br label %bb4

bb3:                                              ; preds = %bb1, %bb
  %9 = add nsw i32 %r.19, 1
  br label %bb4

bb4:                                              ; preds = %bb3, %bb2
  %r.0 = phi i32 [ %9, %bb3 ], [ %8, %bb2 ]
  %10 = add nsw i32 %n.08, 1
  %exitcond = icmp eq i32 %10, %tsets
  br i1 %exitcond, label %bb6, label %bb

bb6:                                              ; preds = %bb4, %entry
  %r.1.lcssa = phi i32 [ 0, %entry ], [ %r.0, %bb4 ]
  ret i32 %r.1.lcssa
}
