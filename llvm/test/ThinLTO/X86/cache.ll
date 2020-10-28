; NetBSD: noatime mounts currently inhibit 'touch -a' updates
; UNSUPPORTED: system-netbsd

; The .noindex suffix for output dir is to prevent Spotlight on macOS from
; indexing it.

; Verify first that *without* hash, we don't use the cache.
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.bc

; Verify that enabling caching is ignoring module without hash
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex
; RUN: ls %t.cache.noindex/llvmcache.timestamp
; RUN: ls %t.cache.noindex | count 1

; Verify that enabling caching is ignoring module without hash with llvm-lto2
; RUN: rm -Rf %t.cache.noindex
; RUN: llvm-lto2 run -o %t.o %t2.bc  %t.bc -cache-dir %t.cache.noindex \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls %t.cache.noindex | count 0


; Repeat again, *with* hash this time.

; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.bc

; Verify that enabling caching is working, and that the pruner only removes
; files matching the pattern "llvmcache-*".
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache-foo %t.cache.noindex/foo
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex
; RUN: ls %t.cache.noindex | count 4
; RUN: ls %t.cache.noindex/llvmcache.timestamp
; RUN: ls %t.cache.noindex/foo
; RUN: not ls %t.cache.noindex/llvmcache-foo
; RUN: ls %t.cache.noindex/llvmcache-* | count 2

; Verify that enabling caching is working with llvm-lto2
; RUN: rm -Rf %t.cache.noindex
; RUN: llvm-lto2 run -o %t.o %t2.bc %t.bc -cache-dir %t.cache.noindex \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls %t.cache.noindex | count 2
; RUN: ls %t.cache.noindex/llvmcache-* | count 2

; Verify that caches with a timestamp older than the pruning interval
; will be pruned
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache-foo
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex
; RUN: not ls %t.cache.noindex/llvmcache-foo

; Verify that specifying a negative number for the pruning interval
; effectively disables the pruning
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache-foo
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex --thinlto-cache-pruning-interval -1
; RUN: ls %t.cache.noindex/llvmcache-foo

; Verify that the pruner doesn't run and a cache file is not deleted when: 
; default values for pruning interval and cache expiration are used, 
; llvmcache.timestamp is current, 
; cache file is older than default cache expiration value.
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache-foo
; RUN: touch %t.cache.noindex/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex
; RUN: ls %t.cache.noindex/llvmcache-foo

; Verify that the pruner runs and a cache file is deleted when:
; pruning interval has value 0 (i.e. run garbage collector now)
; default value for cache expiration is used,
; llvmcache.timestamp is current,
; cache file is older than default cache expiration value.
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: touch -t 197001011200 %t.cache.noindex/llvmcache-foo
; RUN: touch %t.cache.noindex/llvmcache.timestamp
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex --thinlto-cache-pruning-interval 0
; RUN: not ls %t.cache.noindex/llvmcache-foo

; Populate the cache with files with "old" access times, then check llvm-lto updates these file times
; A negative pruning interval is used to avoid removing cache entries
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex
; RUN: touch -a -t 197001011200 %t.cache.noindex/llvmcache-*
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex --thinlto-cache-pruning-interval -1
; RUN: ls -ltu %t.cache.noindex/* | not grep 1970-01-01

; Populate the cache with files with "old" access times, then check llvm-lto2 updates these file times
; RUN: rm -Rf %t.cache.noindex
; RUN: llvm-lto2 run -o %t.o %t2.bc %t.bc -cache-dir %t.cache.noindex \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: touch -a -t 197001011200 %t.cache.noindex/llvmcache-*
; RUN: llvm-lto2 run -o %t.o %t2.bc %t.bc -cache-dir %t.cache.noindex \
; RUN:  -r=%t2.bc,_main,plx \
; RUN:  -r=%t2.bc,_globalfunc,lx \
; RUN:  -r=%t.bc,_globalfunc,plx
; RUN: ls -ltu %t.cache.noindex/* | not grep 1970-01-01

; Verify that specifying max size for the cache directory prunes it to this
; size, removing the oldest files first.
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; Create cache files with different sizes.
; Only 8B and 76B files should stay after pruning.
; RUN: %python -c "with open(r'%t.cache.noindex/llvmcache-foo-100k', 'w') as file: file.truncate(102400)"
; RUN: touch -t 198002011200 %t.cache.noindex/llvmcache-foo-100k
; RUN: %python -c "with open(r'%t.cache.noindex/llvmcache-foo-16', 'w') as file: file.truncate(16)"
; RUN: touch -t 198002021200 %t.cache.noindex/llvmcache-foo-16
; RUN: %python -c "with open(r'%t.cache.noindex/llvmcache-foo-77k', 'w') as file: file.truncate(78848)"
; RUN: touch -t 198002031200 %t.cache.noindex/llvmcache-foo-77k
; RUN: %python -c "with open(r'%t.cache.noindex/llvmcache-foo-8', 'w') as file: file.truncate(8)"
; RUN: touch -t 198002041200 %t.cache.noindex/llvmcache-foo-8
; RUN: %python -c "with open(r'%t.cache.noindex/llvmcache-foo-76', 'w') as file: file.truncate(76)"
; RUN: touch -t 198002051200 %t.cache.noindex/llvmcache-foo-76
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex --thinlto-cache-max-size-bytes 78847 --thinlto-cache-entry-expiration 4294967295
; RUN: ls %t.cache.noindex/llvmcache-foo-8
; RUN: ls %t.cache.noindex/llvmcache-foo-76
; RUN: not ls %t.cache.noindex/llvmcache-foo-16
; RUN: not ls %t.cache.noindex/llvmcache-foo-100k
; RUN: not ls %t.cache.noindex/llvmcache-foo-77k

; Verify that specifying a max size > 4GB for the cache directory does not
; prematurely prune, due to an integer overflow.
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: %python -c "with open(r'%t.cache.noindex/llvmcache-foo-10', 'w') as file: file.truncate(10)"
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex --thinlto-cache-max-size-bytes 4294967297
; RUN: ls %t.cache.noindex/llvmcache-foo-10

; Verify that negative numbers aren't accepted for the
; --thinlto-cache-max-size-bytes switch
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: not llvm-lto %t.bc --thinlto-cache-max-size-bytes -1 2>&1 | FileCheck %s
; CHECK: -thinlto-cache-max-size-bytes option: '-1' value invalid

; Verify that specifying max number of files in the cache directory prunes
; it to this amount, removing the oldest files first.
; RUN: rm -Rf %t.cache.noindex && mkdir %t.cache.noindex
; Create cache files with different sizes.
; Only 75B and 76B files should stay after pruning.
; RUN: %python -c "print(' ' * 1023)" > %t.cache.noindex/llvmcache-foo-1023
; RUN: touch -t 198002011200 %t.cache.noindex/llvmcache-foo-1023
; RUN: %python -c "print(' ' * 15)" > %t.cache.noindex/llvmcache-foo-15
; RUN: touch -t 198002021200 %t.cache.noindex/llvmcache-foo-15
; RUN: %python -c "print(' ' * 7)" > %t.cache.noindex/llvmcache-foo-7
; RUN: touch -t 198002031200 %t.cache.noindex/llvmcache-foo-7
; RUN: %python -c "print(' ' * 75)" > %t.cache.noindex/llvmcache-foo-75
; RUN: touch -t 198002041200 %t.cache.noindex/llvmcache-foo-75
; RUN: %python -c "print(' ' * 76)" > %t.cache.noindex/llvmcache-foo-76
; RUN: touch -t 198002051200 %t.cache.noindex/llvmcache-foo-76
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t2.bc %t.bc -thinlto-cache-dir %t.cache.noindex --thinlto-cache-max-size-files 4 --thinlto-cache-entry-expiration 4294967295
; RUN: ls %t.cache.noindex/llvmcache-foo-75
; RUN: ls %t.cache.noindex/llvmcache-foo-76
; RUN: not ls %t.cache.noindex/llvmcache-foo-15
; RUN: not ls %t.cache.noindex/llvmcache-foo-1024
; RUN: not ls %t.cache.noindex/llvmcache-foo-7

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
