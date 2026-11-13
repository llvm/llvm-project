;; The cache key does not include the module paths, so the result of importing
;; must not depend on them either. Link the same modules twice with the file
;; names of the two imported-from modules swapped, which swaps their path
;; order, and check that the cache entries are identical. The IRMover appends
;; llvm.linker.options in import order.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: mkdir 1 2
; RUN: opt -module-hash -module-summary main.ll -o 1/main.bc
; RUN: opt -module-hash -module-summary foo.ll -o 1/a.bc
; RUN: opt -module-hash -module-summary bar.ll -o 1/b.bc
; RUN: cp 1/main.bc 2/main.bc
; RUN: cp 1/a.bc 2/b.bc
; RUN: cp 1/b.bc 2/a.bc

; RUN: llvm-lto2 run -cache-dir cache1 -o out1 1/main.bc 1/a.bc 1/b.bc \
; RUN:   -r=1/main.bc,_main,plx -r=1/main.bc,_f1,lx -r=1/main.bc,_f2,lx \
; RUN:   -r=1/a.bc,_f1,plx -r=1/b.bc,_f2,plx
; RUN: llvm-lto2 run -cache-dir cache2 -o out2 2/main.bc 2/a.bc 2/b.bc \
; RUN:   -r=2/main.bc,_main,plx -r=2/main.bc,_f1,lx -r=2/main.bc,_f2,lx \
; RUN:   -r=2/b.bc,_f1,plx -r=2/a.bc,_f2,plx
; RUN: ls cache1 | count 3
; RUN: diff -r cache1 cache2

;; Same for the legacy ThinLTOCodeGenerator.
; RUN: mkdir cache3 cache4
; RUN: llvm-lto -thinlto-action=run -exported-symbol=_main \
; RUN:   1/main.bc 1/a.bc 1/b.bc -thinlto-cache-dir cache3
; RUN: llvm-lto -thinlto-action=run -exported-symbol=_main \
; RUN:   2/main.bc 2/a.bc 2/b.bc -thinlto-cache-dir cache4
; RUN: ls cache3 | count 4
; RUN: diff -r cache3 cache4

;--- main.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define void @main() {
  call void @f1()
  call void @f2()
  ret void
}

declare void @f1()
declare void @f2()

;--- foo.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define void @f1() {
  ret void
}

!llvm.linker.options = !{!0}
!0 = !{!"-lfoo"}

;--- bar.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define void @f2() {
  ret void
}

!llvm.linker.options = !{!0}
!0 = !{!"-lbar"}
