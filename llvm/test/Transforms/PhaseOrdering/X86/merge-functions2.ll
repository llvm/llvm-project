; RUN: opt -passes="default<O3>" -enable-merge-functions -S < %s | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx12.0.0"

; Function Attrs: noinline nounwind optsize ssp uwtable
define i32 @f(i32 noundef %x) #0 {
; CHECK-LABEL: @f(
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4, !tbaa !5
  %0 = load i32, ptr %x.addr, align 4, !tbaa !5
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 2, label %sw.bb
    i32 4, label %sw.bb
    i32 6, label %sw.bb
    i32 7, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry, %entry, %entry, %entry, %entry
  store i32 1, ptr %x.addr, align 4, !tbaa !5
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  store i32 0, ptr %x.addr, align 4, !tbaa !5
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb
  %1 = load i32, ptr %x.addr, align 4, !tbaa !5
  ret i32 %1
}

; Function Attrs: noinline nounwind optsize ssp uwtable
define i32 @g(i32 noundef %x) #0 {
; CHECK-LABEL: @g(
; CHECK-NEXT:    [[TMP2:%.*]] = tail call range(i32 0, 2) i32 @f(i32 noundef [[TMP0:%.*]]) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    ret i32 [[TMP2]]
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4, !tbaa !5
  %0 = load i32, ptr %x.addr, align 4, !tbaa !5
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 2, label %sw.bb
    i32 4, label %sw.bb
    i32 6, label %sw.bb
    i32 7, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry, %entry, %entry, %entry, %entry
  store i32 1, ptr %x.addr, align 4, !tbaa !5
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  store i32 0, ptr %x.addr, align 4, !tbaa !5
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb
  %1 = load i32, ptr %x.addr, align 4, !tbaa !5
  ret i32 %1
}

!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
