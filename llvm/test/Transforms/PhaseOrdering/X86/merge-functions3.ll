; RUN: opt -passes="default<O3>" -enable-merge-functions -S < %s | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx12.0.0"

@switch.table.f = private unnamed_addr constant [8 x i32] [i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 1], align 4
@switch.table.g = private unnamed_addr constant [8 x i32] [i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 1], align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none) uwtable
define range(i32 0, 2) i32 @f(i32 noundef %x) local_unnamed_addr #0 {
; CHECK-LABEL: @f(
entry:
  %0 = icmp ult i32 %x, 8
  br i1 %0, label %switch.lookup, label %sw.epilog

switch.lookup:                                    ; preds = %entry
  %1 = zext nneg i32 %x to i64
  %switch.gep = getelementptr inbounds [8 x i32], ptr @switch.table.f, i64 0, i64 %1
  %switch.load = load i32, ptr %switch.gep, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %switch.lookup
  %x.addr.0 = phi i32 [ %switch.load, %switch.lookup ], [ 0, %entry ]
  ret i32 %x.addr.0
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none) uwtable
define range(i32 0, 2) i32 @g(i32 noundef %x) local_unnamed_addr #0 {
; CHECK-LABEL: @g(
; CHECK-NEXT:    [[TMP2:%.*]] = tail call range(i32 0, 2) i32 @f(i32 noundef [[TMP0:%.*]]) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    ret i32 [[TMP2]]
entry:
  %0 = icmp ult i32 %x, 8
  br i1 %0, label %switch.lookup, label %sw.epilog

switch.lookup:                                    ; preds = %entry
  %1 = zext nneg i32 %x to i64
  %switch.gep = getelementptr inbounds [8 x i32], ptr @switch.table.g, i64 0, i64 %1
  %switch.load = load i32, ptr %switch.gep, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %switch.lookup
  %x.addr.0 = phi i32 [ %switch.load, %switch.lookup ], [ 0, %entry ]
  ret i32 %x.addr.0
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind optsize ssp willreturn memory(none) uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cmov,+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
