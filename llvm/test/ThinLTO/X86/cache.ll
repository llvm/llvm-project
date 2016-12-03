; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.bc

; Verify that enabling caching is working
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc  %t.bc -thinlto-cache-dir %t.cache
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache | count 3

; Verify that enabling caching is working with llvm-lto2
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto2 -o %t.o %t2.bc  %t.bc -cache-dir %t.cache \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls %t.cache | count 2

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
