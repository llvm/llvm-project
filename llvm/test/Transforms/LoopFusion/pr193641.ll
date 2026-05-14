; REQUIRES: asserts
; RUN: opt -passes=loop-fusion -loop-fusion-peel-max-count=2 -disable-output -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s
; STAT: 1 loop-fusion - Loops fused

; Test reduced from the file diags_on_lat_aux_grid.F90 present in pop2,
; a program from spec 2017.
; Loop fusion should not crash when flushing the DomTreeUpdater due to
; missed edge removals when deleting basic blocks.

define void @diags_on_lat_aux_grid(i1 %cond, i1 %cond1, i1 %0, ptr noalias %b1, ptr noalias %b2) {
.critedge:
  br i1 %cond, label %common.ret, label %thread-pre-split

common.ret:                                       ; preds = %.loopexit, %.critedge
  ret void

thread-pre-split:                                 ; preds = %.critedge
  br i1 %cond1, label %2, label %.loopexit

2:                                                ; preds = %thread-pre-split
  br i1 %0, label %.lr.ph1333, label %._crit_edge1334

.lr.ph1333:                                       ; preds = %.lr.ph1333, %2
  %indvars.iv1386 = phi i64 [ %indvars.iv.next1387, %.lr.ph1333 ], [ 0, %2 ]
  %3 = getelementptr [8 x i8], ptr %b1, i64 %indvars.iv1386
  store double 0.000000e+00, ptr %3, align 8
  %indvars.iv.next1387 = add i64 %indvars.iv1386, 1
  %exitcond1389.not = icmp eq i64 %indvars.iv1386, 100
  br i1 %exitcond1389.not, label %._crit_edge1334, label %.lr.ph1333

._crit_edge1334:                                  ; preds = %.lr.ph1333, %2
  br i1 %0, label %.lr.ph1336, label %.loopexit

.lr.ph1336:                                       ; preds = %.lr.ph1336, %._crit_edge1334
  %indvars.iv1390 = phi i64 [ %indvars.iv.next1391, %.lr.ph1336 ], [ 0, %._crit_edge1334 ]
  %5 = getelementptr [8 x i8], ptr %b2, i64 %indvars.iv1390
  store double 0.000000e+00, ptr %5, align 8
  %indvars.iv.next1391 = add i64 %indvars.iv1390, 1
  %exitcond1393.not = icmp eq i64 %indvars.iv.next1391, 99
  br i1 %exitcond1393.not, label %.loopexit, label %.lr.ph1336

.loopexit:                                        ; preds = %.lr.ph1336, %._crit_edge1334, %thread-pre-split
  store volatile i32 0, ptr %b1, align 4
  br label %common.ret
}
