; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -module-hash -module-summary %s -o %t/t.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-import-lists3.ll -o %t/a.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-import-lists4.ll -o %t/b.bc

; Tests that the hash for t is insensitive to the order of the modules.

; RUN: llvm-lto2 run -cache-dir %t/cache -o %t.o %t/t.bc %t/a.bc %t/b.bc -r=%t/t.bc,main,plx -r=%t/t.bc,f1,lx -r=%t/t.bc,f2,lx -r=%t/a.bc,f1,plx -r=%t/b.bc,f2,plx
; RUN: ls %t/cache | count 3

; RUN: llvm-lto2 run -cache-dir %t/cache -o %t.o %t/b.bc %t/a.bc %t/t.bc -r=%t/t.bc,main,plx -r=%t/t.bc,f1,lx -r=%t/t.bc,f2,lx -r=%t/a.bc,f1,plx -r=%t/b.bc,f2,plx
; RUN: ls %t/cache | count 3

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  call void @f1()
  call void @f2()
  ret void
}

declare void @f1()
declare void @f2()
