; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %p/Inputs/cache.ll -o %t2.bc

; Check that the generating object files is working without cache
; RUN: rm -Rf %t.thin.out
; RUN: llvm-lto -thinlto-save-objects=%t.thin.out -thinlto-action=run %t2.bc  %t.bc -exported-symbol=main 
; RUN: ls %t.thin.out | count 2

; Same with cache
; RUN: rm -Rf %t.thin.out
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-save-objects=%t.thin.out -thinlto-action=run %t2.bc  %t.bc -exported-symbol=main -thinlto-cache-dir %t.cache 
; RUN: ls %t.thin.out | count 2
; RUN: ls %t.cache | count 3

; Same with hot cache
; RUN: rm -Rf %t.thin.out
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto -thinlto-save-objects=%t.thin.out -thinlto-action=run %t2.bc  %t.bc -exported-symbol=main -thinlto-cache-dir %t.cache 
; RUN: ls %t.thin.out | count 2
; RUN: ls %t.cache | count 3

; Check the name of object in directory has arch name included.
; RUN: ls %t.thin.out | grep x86_64.thinlto.o | count 2


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
