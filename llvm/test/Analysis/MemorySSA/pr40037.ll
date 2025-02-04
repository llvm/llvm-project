; REQUIRES: asserts
; RUN: opt -S -passes="loop-mssa(simple-loop-unswitch)" -verify-memoryssa  < %s | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

; CHECK-LABEL: @func_23()
define dso_local void @func_23() local_unnamed_addr {
  br label %bb2
bb2:                                              ; preds = %bb2.loopexit, %bb
  br label %bb7
bb3.bb9.preheader_crit_edge:                      ; preds = %bb7
  br label %bb9.preheader
bb9.preheader:                                    ; preds = %bb3.bb9.preheader_crit_edge, %bb2
  %tmp20 = icmp eq i32 0, 0
  %tmp24 = icmp ugt i32 65536, 65535
  br label %bb13
bb7:                                              ; preds = %bb7.lr.ph, %bb7
  %tmp6 = icmp slt i8 94, 6
  br i1 %tmp6, label %bb7, label %bb3.bb9.preheader_crit_edge
bb9:                                              ; preds = %bb21
  store i16 %tmp27, ptr undef, align 2
  %tmp12 = icmp eq i16 %tmp27, 1
  br i1 %tmp12, label %bb28, label %bb13
bb13:                                             ; preds = %bb9.preheader, %bb9
  %storemerge3 = phi i16 [ 0, %bb9.preheader ], [ %tmp27, %bb9 ]
  br i1 %tmp20, label %bb21, label %bb28
bb21:                                             ; preds = %bb13
  %tmp27 = add i16 %storemerge3, 1
  br i1 %tmp24, label %bb2, label %bb9
bb28:                                             ; preds = %bb13, %bb9
  ret void
}
