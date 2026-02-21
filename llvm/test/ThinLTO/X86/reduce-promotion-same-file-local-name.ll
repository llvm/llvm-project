; Test a simgple cross module promotion where two same-name same-file static functions
; are in both modules respectively.

; RUN: opt -thinlto-bc %s -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/reduce-promotion-same-file-local-name.ll -o %t2.bc
;
; RUN: llvm-lto2 run %t1.bc %t2.bc \
; RUN:  --whole-program-visibility-enabled-in-lto=true \
; RUN:  -disable-always-rename-promoted-locals \
; RUN:  -save-temps -o %t3 \
; RUN:  -r %t1.bc,m1,px \
; RUN:  -r %t2.bc,m2,p \
; RUN:  -r %t1.bc,m2,x
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT1
; RUN: llvm-dis %t3.2.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT2
;
; IMPORT1:      define dso_local i32 @m1(
; IMPORT1:        tail call fastcc i32 @foo.{{[0-9]+}}(
; IMPORT1:      define available_externally i32 @m2(
; IMPORT1-NEXT:   %2 = tail call fastcc i32 @foo(
; IMPORT2:      define hidden fastcc range(i32 -2147483647, -2147483648) i32 @foo(

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

source_filename = "foo.c"

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
