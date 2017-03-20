; Verify first that *without* hash, we don't use the cache.

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.bc

; Verify that enabling caching is ignoring module without hash
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc  %t.bc -thinlto-cache-dir %t.cache
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache | count 1

; Verify that enabling caching is ignoring module without hash with llvm-lto2
; RUN: rm -Rf %t.cache
; RUN: llvm-lto2 -o %t.o %t2.bc  %t.bc -cache-dir %t.cache \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls %t.cache | count 0


; Repeat again, *with* hash this time.

; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.bc

; Verify that enabling caching is working, and that the pruner only removes
; files matching the pattern "llvmcache-*".
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo %t.cache/foo
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc  %t.bc -thinlto-cache-dir %t.cache
; RUN: ls %t.cache | count 4
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache/foo
; RUN: not ls %t.cache/llvmcache-foo
; RUN: ls %t.cache/llvmcache-* | count 2

; Verify that enabling caching is working with llvm-lto2
; RUN: rm -Rf %t.cache
; RUN: llvm-lto2 -o %t.o %t2.bc  %t.bc -cache-dir %t.cache \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls %t.cache | count 2
; RUN: ls %t.cache/llvmcache-* | count 2

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
