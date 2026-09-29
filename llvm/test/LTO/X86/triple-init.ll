; Test to ensure that the LTO pipelines add pass to build the TargetLibraryInfo
; using the specified target triple.

; Check with regular LTO
; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -exported-symbol=_main  -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s
; Check with ThinLTO. Use llvm-lto2 since this adds earlier passes requiring
; the TargetLibraryInfo with ThinLTO (WholeProgramDevirt).
; RUN: opt -module-summary -o %t1 %s
; RUN: llvm-lto2 run -r %t1,_pow, -r %t1,_main,plx -o %t2 %t1
; RUN: llvm-nm %t2.1 | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

declare double @pow(double, double)

define double @main(double %x) {
; We check that LTO will be aware of target triple and apply pow to __exp10 transformation.
; CHECK: U ___exp10
  %retval = call double @pow(double 10.0, double %x)
  ret double %retval
}
