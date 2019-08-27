; RUN: opt -loop-rotate -licm -enable-mssa-loop-dependency -verify-memoryssa %s -S | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; CHECK-LABEL: @func_42()
define void @func_42() {
entry:
  br label %for.cond1050

for.cond1050.loopexit:                            ; preds = %for.cond1373
  br label %for.cond1050

for.cond1050:                                     ; preds = %for.cond1050.loopexit, %entry
  %storemerge6 = phi i32 [ 2, %entry ], [ 0, %for.cond1050.loopexit ]
  %cmp1051 = icmp sgt i32 %storemerge6, -1
  br i1 %cmp1051, label %for.cond1055.preheader, label %cleanup1400.loopexit1

for.cond1055.preheader:                           ; preds = %for.cond1050
  store i64 0, i64* null, align 8
  %0 = load i64, i64* null, align 8
  %tobool1383 = icmp eq i64 %0, 0
  br i1 %tobool1383, label %for.cond1055.preheader.cleanup1400.loopexit.split_crit_edge, label %for.cond1055.preheader.for.cond1055.preheader.split_crit_edge

for.cond1055.preheader.for.cond1055.preheader.split_crit_edge: ; preds = %for.cond1055.preheader
  br label %for.body1376

for.cond1055.preheader.cleanup1400.loopexit.split_crit_edge: ; preds = %for.cond1055.preheader
  br label %cleanup1400.loopexit.split

for.cond1373:                                     ; preds = %for.body1376
  br i1 true, label %for.body1376, label %for.cond1050.loopexit

for.body1376:                                     ; preds = %for.cond1373, %for.cond1055.preheader.for.cond1055.preheader.split_crit_edge
  br i1 false, label %cleanup1400.loopexit, label %for.cond1373

cleanup1400.loopexit:                             ; preds = %for.body1376
  br label %cleanup1400.loopexit.split

cleanup1400.loopexit.split:                       ; preds = %cleanup1400.loopexit, %for.cond1055.preheader.cleanup1400.loopexit.split_crit_edge
  br label %cleanup1400

cleanup1400.loopexit1:                            ; preds = %for.cond1050
  br label %cleanup1400

cleanup1400:                                      ; preds = %cleanup1400.loopexit1, %cleanup1400.loopexit.split
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull undef)
  unreachable
}
