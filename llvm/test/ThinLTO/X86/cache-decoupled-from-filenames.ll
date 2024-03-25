; RUN: rm -rf %t && mkdir -p %t/1 %t/2 %t/3 %t/4
; RUN: opt -module-hash -module-summary %s -o %t/t.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-import-lists1.ll -o %t/1/a.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-import-lists2.ll -o %t/2/b.bc

; Tests that the hash for t is insensitive to the bitcode module filenames.

; RUN: rm -rf %t/cache
; RUN: llvm-lto2 run -cache-dir %t/cache -o %t.o %t/t.bc %t/1/a.bc %t/2/b.bc -r=%t/t.bc,main,plx -r=%t/t.bc,f1,lx -r=%t/t.bc,f2,lx -r=%t/1/a.bc,f1,plx -r=%t/1/a.bc,linkonce_odr,plx -r=%t/2/b.bc,f2,plx -r=%t/2/b.bc,linkonce_odr,lx
; RUN: ls %t/cache | count 3

; RUN: cp %t/1/a.bc %t/4/d.bc
; RUN: cp %t/2/b.bc %t/3/k.bc
; RUN: llvm-lto2 run -cache-dir %t/cache -o %t.o %t/t.bc %t/4/d.bc %t/3/k.bc -r=%t/t.bc,main,plx -r=%t/t.bc,f1,lx -r=%t/t.bc,f2,lx -r=%t/4/d.bc,f1,plx -r=%t/4/d.bc,linkonce_odr,plx -r=%t/3/k.bc,f2,plx -r=%t/3/k.bc,linkonce_odr,lx
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
