; RUN: opt -module-summary %s -S -module-summary | FileCheck %s
; CHECK-NOT: typeidMayByAccessed
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef i32 @_Z4testP1Ai(ptr noundef readnone captures(none) %0, i32 noundef %1) local_unnamed_addr {
  %3 = and i32 %1, 1
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %9, label %5

5:                                                ; preds = %2
  %6 = tail call noundef ptr @_Z7createBv()
  %7 = load ptr, ptr %6, align 8
  %8 = tail call i1 @llvm.public.type.test(ptr %7, metadata !"_ZTS1B")
  tail call void @llvm.assume(i1 %8)
  br label %13

9:                                                ; preds = %2
  %10 = tail call noundef ptr @_Z7createCv()
  %11 = load ptr, ptr %10, align 8
  %12 = tail call i1 @llvm.public.type.test(ptr %11, metadata !"_ZTS1C")
  tail call void @llvm.assume(i1 %12)
  br label %13

13:                                               ; preds = %9, %5
  %15 = phi ptr [ %11, %9 ], [ %7, %5 ]
  %16 = phi ptr [ %10, %9 ], [ %6, %5 ]
  
  %17 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %18 = load ptr, ptr %17, align 8
  %19 = tail call noundef i32 %18(ptr noundef nonnull align 8 dereferenceable(8) %16)
  ret i32 %19
}

declare noundef ptr @_Z7createBv() local_unnamed_addr

declare i1 @llvm.public.type.test(ptr, metadata)

declare void @llvm.assume(i1 noundef)

declare noundef ptr @_Z7createCv() local_unnamed_addr
