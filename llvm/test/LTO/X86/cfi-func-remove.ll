; RUN: opt -thinlto-bc -thinlto-split-lto-unit -unified-lto <%s -o %t0
; RUN: llvm-lto2 run -o %t1 --unified-lto=full --save-temps %t0
; RUN: llvm-dis <%t1.0.0.preopt.bc 2>&1 | FileCheck %s --implicit-check-not warning:
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

!cfi.functions = !{!2}
; CHECK-NOT: cfi.functions

!2 = !{!"main", i8 0}
