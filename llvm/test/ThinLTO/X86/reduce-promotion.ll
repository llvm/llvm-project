; Test a simgple cross module promotion where the suffix '.llvm.<...>' is not needed.

; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t

; RUN: opt -thinlto-bc %t/a.ll -o %t/a.bc
; RUN: opt -thinlto-bc %t/b.ll -o %t/b.bc
;
; RUN: llvm-lto2 run %t/a.bc %t/b.bc \
; RUN:  --whole-program-visibility-enabled-in-lto=true \
; RUN:  -disable-always-rename-promoted-locals \
; RUN:  -save-temps -o %t/lto-out \
; RUN:  -r %t/a.bc,m1,px \
; RUN:  -r %t/b.bc,m2,p \
; RUN:  -r %t/a.bc,m2,x
; RUN: llvm-dis %t/lto-out.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT1
; RUN: llvm-dis %t/lto-out.2.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT2
;
; IMPORT1:      define available_externally i32 @m2(
; IMPORT1-NEXT:   %2 = tail call fastcc i32 @foo(
; IMPORT2:      define hidden fastcc range(i32 -2147483647, -2147483648) i32 @foo(

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @m1(i32 noundef %0) local_unnamed_addr {
  %2 = tail call i32 @m2(i32 noundef %0)
  ret i32 %2
}

declare i32 @m2(i32 noundef) local_unnamed_addr

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

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
