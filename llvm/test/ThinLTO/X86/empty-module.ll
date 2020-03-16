; RUN: opt -module-summary -o %t.bc %s

; RUN: rm -f %t2.0
; RUN: llvm-lto2 run  %t.bc -r %t.bc,foo,pl -o %t2 -thinlto-distributed-indexes
; RUN: llvm-readobj -h %t2.0 | FileCheck %s
; RUN: llvm-nm %t2.0 2>&1 | count 0

; CHECK: Format: elf64-x86-64

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), i64 ()* @foo_ifunc

define internal i64 @foo_ifunc() {
entry:
  ret i64 0
}
