; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-import-lists1.ll -o %t1.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-import-lists2.ll -o %t2.bc

; Tests that the hash for t is sensitive to the set of imported functions
; for each module, which in this case depends on the link order (the function
; linkonce_odr will be imported from either t1 or t2, whichever comes first).

; RUN: rm -rf %t.cache
; RUN: llvm-lto2 -cache-dir %t.cache -o %t.o %t.bc %t1.bc %t2.bc -r=%t.bc,main,plx -r=%t.bc,f1,lx -r=%t.bc,f2,lx -r=%t1.bc,f1,plx -r=%t1.bc,linkonce_odr,plx -r=%t2.bc,f2,plx -r=%t2.bc,linkonce_odr,lx
; RUN: llvm-lto2 -cache-dir %t.cache -o %t.o %t.bc %t2.bc %t1.bc -r=%t.bc,main,plx -r=%t.bc,f1,lx -r=%t.bc,f2,lx -r=%t2.bc,f2,plx -r=%t2.bc,linkonce_odr,plx -r=%t1.bc,f1,plx -r=%t1.bc,linkonce_odr,lx
; RUN: ls %t.cache | count 6

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  call void @f1()
  call void @f2()
  ret void
}

declare void @f1()
declare void @f2()
