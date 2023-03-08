; No hash produced, empty file module entry.
; RUN: opt -module-summary %s -o %t.bc

; RUN: rm -rf %t.cache.noindex && mkdir %t.cache.noindex
; RUN: llvm-lto -thinlto-action=run -exported-symbol=globalfunc %t.bc \
; RUN:   -thinlto-cache-dir %t.cache.noindex -thinlto-save-objects %t.save.noindex | FileCheck %s --allow-empty

; CHECK-NOT: remarks

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc() #0 {
entry:
  ret void
}
