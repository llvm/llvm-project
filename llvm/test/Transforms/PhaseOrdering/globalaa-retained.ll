; RUN: opt -O3 -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

@v = internal unnamed_addr global i32 0, align 4
@p = common global i32* null, align 8

; Function Attrs: norecurse nounwind
define void @f(i32 %n) {
entry:
  %0 = load i32, i32* @v, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @v, align 4
  %1 = load i32*, i32** @p, align 8
  store i32 %n, i32* %1, align 4
  %2 = load i32, i32* @v, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, i32* @v, align 4
  ret void
}

; check variable v is loaded only once after optimization, which should be
; prove that globalsAA survives until the optimization that can use it to
; optimize away the duplicate load/stores on variable v.
; CHECK:     load i32, i32* @v, align 4
; CHECK-NOT: load i32, i32* @v, align 4
