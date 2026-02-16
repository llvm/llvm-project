; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t

; RUN: opt -thinlto-bc %t/a.ll -o %t/a.bc
; RUN: opt -thinlto-bc %t/b.ll -o %t/b.bc
;
; RUN: llvm-lto2 run %t/a.bc %t/b.bc \
; RUN:   -thinlto-distributed-indexes \
; RUN:  --whole-program-visibility-enabled-in-lto=true \
; RUN:  -force-import-all \
; RUN:  -save-temps -o %t/lto-out \
; RUN:  -r %t/a.bc,m1,px \
; RUN:  -r %t/b.bc,m2,p \
; RUN:  -r %t/a.bc,m2,x

; RUN: opt -passes=function-import -import-all-index -enable-import-metadata -summary-file %t/a.bc.thinlto.bc %t/a.bc -o %t/a.bc.out
; RUN: opt -passes=function-import -import-all-index -summary-file %t/b.bc.thinlto.bc %t/b.bc -o %t/b.bc.out
; RUN: llvm-dis %t/a.bc.out -o - | FileCheck %s --check-prefix=CHECK-A
; RUN: llvm-dis %t/b.bc.out -o - | FileCheck %s --check-prefix=CHECK-B

; CHECK-A: define hidden fastcc range(i32 -2147483647, -2147483648) i32 @foo.llvm.
; CHECK-A: define available_externally hidden fastcc range(i32 -2147483647, -2147483648) i32 @foo(
; CHECK-B: define hidden fastcc range(i32 -2147483647, -2147483648) i32 @foo(

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @m1(i32 noundef %0) local_unnamed_addr {
  %2 = shl nsw i32 %0, 1
  %3 = tail call fastcc i32 @foo(i32 noundef %2)
  %4 = tail call i32 @m2(i32 noundef %0)
  %5 = add nsw i32 %4, %3
  ret i32 %5
}

define internal fastcc range(i32 -2147483647, -2147483648) i32 @foo(i32 noundef %0) unnamed_addr #1 {
  %2 = add nsw i32 %0, 5
  %3 = sdiv i32 %2, %0
  ret i32 %3
}

declare i32 @m2(i32 noundef) local_unnamed_addr

attributes #1 = { noinline }

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @m2(i32 noundef %0) local_unnamed_addr {
  %2 = tail call fastcc i32 @foo(i32 noundef %0)
  %3 = shl nsw i32 %0, 1
  %4 = tail call fastcc i32 @foo(i32 noundef %3)
  %5 = add nsw i32 %4, %2
  ret i32 %5
}

define internal fastcc range(i32 -2147483647, -2147483648) i32 @foo(i32 noundef %0) unnamed_addr #1 {
  %2 = add nsw i32 %0, 5
  %3 = sdiv i32 %2, %0
  ret i32 %3
}

attributes #1 = { noinline }
