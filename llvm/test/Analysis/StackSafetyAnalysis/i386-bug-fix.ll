; REQUIRES: x86-registered-target

; RUN: opt -passes="print-stack-safety" -disable-output %s 2>&1 | FileCheck %s --check-prefixes=CHECK

; CHECK:      @main
; CHECK-NEXT:   args uses:
; CHECK-NEXT:     argv[]: empty-set
; CHECK-NEXT:   allocas uses:
; CHECK-NEXT:     [4]: [0,4)
; CHECK-NEXT:     [32]: full-set
; CHECK-NEXT:   safe accesses:

target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

; Function Attrs: mustprogress norecurse sanitize_address uwtable
define dso_local i32 @main(i32 %argc, ptr %argv)  {
entry:
  %0 = alloca i32, align 4
  %1 = alloca i8, i64 32, align 32
  %2 = ptrtoint ptr %1 to i32
  store i32 %2, ptr %0, align 4
  ret i32 0
}
