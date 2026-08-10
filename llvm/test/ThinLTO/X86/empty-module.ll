; RUN: opt -module-summary -o %t.bc %s

; RUN: rm -f %t2.*
; RUN: llvm-lto2 run %t.bc -r %t.bc,foo,pl -o %t2 -thinlto-distributed-indexes -save-temps
; Ensure lto does not emit empty combined module.
; RUN: test ! -e %t2.0
; Ensure empty combined module has only 2 temp files.
; RUN: ls %t2.0.*.bc | count 2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), ptr @foo_ifunc

define internal ptr @foo_ifunc() {
entry:
  ret ptr null
}
