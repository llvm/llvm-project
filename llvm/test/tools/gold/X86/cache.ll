; Verify that enabling caching is ignoring module when we emit them without hash.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/cache.ll -o %t2.o

; RUN: rm -Rf %t.cache
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     -o %t3.o %t2.o %t.o

; Since nothing was added to the cache, there shouldn't be a timestamp file yet.
; RUN: not ls %t.cache


; Verify that enabling caching is working with module with hash.

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.o

; RUN: rm -Rf %t.cache
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     -o %t3.o %t2.o %t.o

; Two cached objects, plus a timestamp file
; RUN: ls %t.cache | count 3


; Create two files that would be removed by cache pruning due to age.
; We should only remove files matching the pattern "llvmcache-*".

; RUN: touch -t 197001011200 %t.cache/llvmcache-foo %t.cache/foo
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     --plugin-opt=cache-policy=prune_after=1h:prune_interval=0s \
; RUN:     -o %t3.o %t2.o %t.o

; Two cached objects, plus a timestamp file and "foo", minus the file we removed.
; RUN: ls %t.cache | count 4


; Create a file of size 64KB.
; RUN: %python -c "print(' ' * 65536)" > %t.cache/llvmcache-foo

; This should leave the file in place.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     --plugin-opt=cache-policy=cache_size_bytes=128k:prune_interval=0s \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: ls %t.cache | count 5


; Increase the age of llvmcache-foo
; RUN: touch -r %t.cache/llvmcache-foo -d '-2 minutes' %t.cache/llvmcache-foo

; This should remove it.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=save-temps \
; RUN:     --plugin-opt=cache-dir=%t.cache \
; RUN:     --plugin-opt=cache-policy=cache_size_bytes=32k:prune_interval=0s \
; RUN:     -o %t4.o %t2.o %t.o
; RUN: ls %t.cache | count 4
; With save-temps we can confirm that the cached files were copied into temp
; files to avoid a race condition with the cached files being pruned, since the
; gold plugin-api only accepts native objects passed back as files.
; RUN: ls %t4.o.lto.o1 %t4.o.lto.o2


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() #0 {
entry:
  ret void
}
