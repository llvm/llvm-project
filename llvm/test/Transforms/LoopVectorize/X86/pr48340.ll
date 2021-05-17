; RUN: opt -loop-vectorize --force-vector-width=4 --force-vector-interleave=0 -S -o - < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { i32 }
%1 = type { i64 }

define void @foo(i64* %p, i64* %p.last) unnamed_addr #0 {
; CHECK-LABEL: @foo(
; CHECK: vector.body:
; CHECK:         [[WIDE_MASKED_GATHER0:%.*]] = call <4 x %0*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.0(<4 x %0**> [[TMP5:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %0*> undef)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER1:%.*]] = call <4 x %0*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.0(<4 x %0**> [[TMP6:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %0*> undef)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER2:%.*]] = call <4 x %0*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.0(<4 x %0**> [[TMP7:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %0*> undef)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER3:%.*]] = call <4 x %0*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.0(<4 x %0**> [[TMP8:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %0*> undef)
entry:
  br label %loop

loop:
  %p2 = phi i64* [ %p, %entry ], [ %p.inc, %loop ]
  %p.inc = getelementptr inbounds i64, i64* %p2, i64 2
  %p3 = bitcast i64* %p2 to %0**
  %v = load %0*, %0** %p3, align 8
  %b = icmp eq i64* %p.inc, %p.last
  br i1 %b, label %exit, label %loop

exit:
  ret void
}

define void @bar(i64* %p, i64* %p.last) unnamed_addr #0 {
; CHECK-LABEL: @bar(
; CHECK: vector.body:
; CHECK:         [[WIDE_MASKED_GATHER0:%.*]] = call <4 x %1*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.1(<4 x %1**> [[TMP5:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %1*> undef)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER1:%.*]] = call <4 x %1*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.1(<4 x %1**> [[TMP6:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %1*> undef)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER2:%.*]] = call <4 x %1*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.1(<4 x %1**> [[TMP7:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %1*> undef)
; CHECK-NEXT:    [[WIDE_MASKED_GATHER3:%.*]] = call <4 x %1*> @llvm.masked.gather.v4p0s_s.v4p0p0s_s.1(<4 x %1**> [[TMP8:%.*]], i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x %1*> undef)
entry:
  br label %loop

loop:
  %p2 = phi i64* [ %p, %entry ], [ %p.inc, %loop ]
  %p.inc = getelementptr inbounds i64, i64* %p2, i64 2
  %p3 = bitcast i64* %p2 to %1**
  %v = load %1*, %1** %p3, align 8
  %b = icmp eq i64* %p.inc, %p.last
  br i1 %b, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { "target-cpu"="skylake" }

