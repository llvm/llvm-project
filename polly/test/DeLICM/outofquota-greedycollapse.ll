; RUN: opt %loadNPMPolly '-passes=polly-custom<delicm>' -polly-process-unprofitable -pass-remarks-analysis=polly-delicm -disable-output < %s 2>&1 | FileCheck %s

; This test exercises the ISL operations guard in the DeLICM phase when
; collapsing scalars to a store. The structure of the SCoP below has a few
; important characteristics:
; a) A large loop iteration space.
; b) A long if-else ladder such that, of the two branches exiting each
;    conditional block, one is always a predecessor of a common basic block.
; c) The domain of the common BB is the union of all domains reaching it,
;    which forces certain ISL operations to handle a high-dimensional,
;    extremely large search space.
; Thus it is necessary to bound these operations and bail out instead of
; hanging indefinitely.
; With the shared IslMaxOperationsGuard[budget defined by -polly-delicm-max-ops]
; now armed during collapseScalarsToStore() as well, the analysis aborts
; once the operations budget is exhausted, emitting an "OutOfQuota" remark
; and leaving the SCoP unmodified.

; CHECK: maximal number of operations exceeded during collapseScalarsToStore

define void @ham(ptr %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb19, %bb
  %phi = phi i64 [ 1, %bb ], [ %add, %bb19 ]
  %trunc = trunc i64 %phi to i32
  %and = and i32 %trunc, 1
  %icmp = icmp eq i32 %and, 0
  br i1 %icmp, label %bb4, label %bb2

bb2:                                              ; preds = %bb16, %bb13, %bb10, %bb7, %bb4, %bb1
  %phi3 = phi i8 [ 1, %bb1 ], [ 0, %bb4 ], [ 1, %bb13 ], [ 0, %bb16 ], [ 0, %bb10 ], [ 1, %bb7 ]
  %getelementptr = getelementptr i8, ptr %arg, i64 %phi
  store i8 %phi3, ptr %getelementptr, align 1
  br label %bb19

bb4:                                              ; preds = %bb1
  %and5 = and i64 %phi, 254
  %icmp6 = icmp eq i64 %and5, 0
  br i1 %icmp6, label %bb7, label %bb2

bb7:                                              ; preds = %bb4
  %and8 = and i32 %trunc, 256
  %icmp9 = icmp eq i32 %and8, 0
  br i1 %icmp9, label %bb10, label %bb2

bb10:                                             ; preds = %bb7
  %and11 = and i64 %phi, 3584
  %icmp12 = icmp eq i64 %and11, 0
  br i1 %icmp12, label %bb13, label %bb2

bb13:                                             ; preds = %bb10
  %and14 = and i32 %trunc, 8192
  %icmp15 = icmp eq i32 %and14, 0
  br i1 %icmp15, label %bb16, label %bb2

bb16:                                             ; preds = %bb13
  %and17 = and i64 %phi, 49152
  %icmp18 = icmp eq i64 %and17, 0
  br i1 %icmp18, label %bb19, label %bb2

bb19:                                             ; preds = %bb16, %bb2
  %add = add i64 %phi, 1
  %icmp20 = icmp eq i64 %add, 65536
  br i1 %icmp20, label %bb21, label %bb1

bb21:                                             ; preds = %bb19
  ret void
}
