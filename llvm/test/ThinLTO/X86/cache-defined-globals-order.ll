; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -module-hash -module-summary %s -o %t/t.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-defined-globals-order.ll -o %t/a.bc

; Tests that the LTO cache key is insensitive to the order of the modules.
; The linkonce_odr function @shared29 is defined in both t.bc and a.bc, so
; it gets a different position in the combined index depending on which
; module is processed first, which results in a different insertion order
; for DefinedGlobals.

; RUN: llvm-lto2 run -cache-dir %t/cache -o %t.o %t/t.bc %t/a.bc -r=%t/t.bc,main,plx -r=%t/t.bc,shared29,plx -r=%t/a.bc,shared29,lx
; RUN: ls %t/cache | count 2

; RUN: llvm-lto2 run -cache-dir %t/cache -o %t.o %t/a.bc %t/t.bc -r=%t/t.bc,main,plx -r=%t/t.bc,shared29,plx -r=%t/a.bc,shared29,lx
; RUN: ls %t/cache | count 2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  call void @shared29()
  ret void
}

define linkonce_odr void @shared29() {
  ret void
}
