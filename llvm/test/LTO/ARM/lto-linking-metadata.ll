; RUN: opt %s -o %t1.bc

; RUN: llvm-lto %t1.bc -o %t1.save.opt -save-merged-module -O1 --exported-symbol=foo
; RUN: llvm-dis < %t1.save.opt.merged.bc | FileCheck %s

; RUN: llvm-lto2 run %t1.bc -o %t.out.o -save-temps \
; RUN:     -r=%t1.bc,foo,pxl
; RUN: llvm-dis < %t.out.o.0.2.internalize.bc | FileCheck  %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7a-unknown-linux"

define void @foo() {
entry:
  ret void
}

; CHECK: !llvm.module.flags = !{[[MD_NUM:![0-9]+]]}
; CHECK: [[MD_NUM]] = !{i32 1, !"LTOPostLink", i32 1}
