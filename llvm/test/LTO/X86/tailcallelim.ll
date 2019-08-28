; Check that the LTO pipelines add the Tail Call Elimination pass.

; RUN: llvm-as < %s > %t1
; RUN: llvm-lto -o %t2 %t1 --exported-symbol=foo -save-merged-module
; RUN: llvm-dis < %t2.merged.bc | FileCheck %s

; RUN: llvm-lto2 run -r %t1,foo,plx -r %t1,bar,plx -o %t3 %t1 -save-temps
; RUN: llvm-dis < %t3.0.4.opt.bc | FileCheck %s

; RUN: llvm-lto2 run -r %t1,foo,plx -r %t1,bar,plx -o %t4 %t1 -save-temps -use-new-pm
; RUN: llvm-dis < %t4.0.4.opt.bc | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
; CHECK: tail call void @bar()
  call void @bar()
  ret void
}

declare void @bar()
