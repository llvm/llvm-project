; Verify first that *without* hash, we don't use the cache.

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.bc

; Verify that enabling caching is ignoring module without hash
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache | count 1

; Verify that enabling caching is ignoring module without hash with llvm-lto2
; RUN: rm -Rf %t.cache
; RUN: llvm-lto2 run -o %t.o %t2.bc  %t.bc -cache-dir %t.cache \
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
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache
; RUN: ls %t.cache | count 4
; RUN: ls %t.cache/llvmcache.timestamp
; RUN: ls %t.cache/foo
; RUN: not ls %t.cache/llvmcache-foo
; RUN: ls %t.cache/llvmcache-* | count 2

; Verify that enabling caching is working with llvm-lto2
; RUN: rm -Rf %t.cache
; RUN: llvm-lto2 run -o %t.o %t2.bc %t.bc -cache-dir %t.cache \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls %t.cache | count 2
; RUN: ls %t.cache/llvmcache-* | count 2

; Verify that caches with a timestamp older than the pruning interval
; will be pruned
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo
; RUN: touch -t 197001011200 %t.cache/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache
; RUN: not ls %t.cache/llvmcache-foo

; Verify that specifying a negative number for the pruning interval
; effectively disables the pruning
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo
; RUN: touch -t 197001011200 %t.cache/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache --thinlto-cache-pruning-interval -1
; RUN: ls %t.cache/llvmcache-foo

; Verify that the pruner doesn't run and a cache file is not deleted when: 
; default values for pruning interval and cache expiration are used, 
; llvmcache.timestamp is current, 
; cache file is older than default cache expiration value.
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo
; RUN: touch %t.cache/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache
; RUN: ls %t.cache/llvmcache-foo

; Verify that the pruner runs and a cache file is deleted when:
; pruning interval has value 0 (i.e. run garbage collector now)
; default value for cache expiration is used,
; llvmcache.timestamp is current,
; cache file is older than default cache expiration value.
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: touch -t 197001011200 %t.cache/llvmcache-foo
; RUN: touch %t.cache/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache --thinlto-cache-pruning-interval 0
; RUN: not ls %t.cache/llvmcache-foo

; Verify that specifying max size for the cache directory prunes it to this
; size, removing the largest files first.
; RUN: rm -Rf %t.cache && mkdir %t.cache
; Create cache files with different sizes.
; Only 8B, 16B and 76B files should stay after pruning.
; RUN: %python -c "print(' ' * 1023)" > %t.cache/llvmcache-foo-1024
; RUN: %python -c "print(' ' * 15)" > %t.cache/llvmcache-foo-16
; RUN: %python -c "print(' ' * 7)" > %t.cache/llvmcache-foo-8
; RUN: %python -c "print(' ' * 75)" > %t.cache/llvmcache-foo-76
; RUN: %python -c "print(' ' * 76)" > %t.cache/llvmcache-foo-77
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache --thinlto-cache-max-size-bytes 100
; RUN: ls %t.cache/llvmcache-foo-16
; RUN: ls %t.cache/llvmcache-foo-8
; RUN: ls %t.cache/llvmcache-foo-76
; RUN: not ls %t.cache/llvmcache-foo-1024
; RUN: not ls %t.cache/llvmcache-foo-77

; Verify that specifying max number of files in the cache directory prunes
; it to this amount, removing the largest files first.
; RUN: rm -Rf %t.cache && mkdir %t.cache
; Create cache files with different sizes.
; Only 8B and 16B files should stay after pruning.
; RUN: %python -c "print(' ' * 1023)" > %t.cache/llvmcache-foo-1024
; RUN: %python -c "print(' ' * 15)" > %t.cache/llvmcache-foo-16
; RUN: %python -c "print(' ' * 7)" > %t.cache/llvmcache-foo-8
; RUN: %python -c "print(' ' * 75)" > %t.cache/llvmcache-foo-76
; RUN: %python -c "print(' ' * 76)" > %t.cache/llvmcache-foo-77
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache --thinlto-cache-max-size-files 2
; RUN: ls %t.cache/llvmcache-foo-16
; RUN: ls %t.cache/llvmcache-foo-8
; RUN: not ls %t.cache/llvmcache-foo-76
; RUN: not ls %t.cache/llvmcache-foo-1024
; RUN: not ls %t.cache/llvmcache-foo-77

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
