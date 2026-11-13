;; Test that with -icp-allow-decls, the ThinLTO backend only promotes an
;; indirect call to a function that is merely declared in the module when the
;; index has a summary for the target indicating that it is live. Otherwise the
;; promoted direct call would be an undefined reference at link time, since
;; dead definitions are dropped in their defining module while declarations of
;; them in other modules are not.

; RUN: split-file %s %t
;; Only the hottest profiled target (bar) gets a call edge in foo's summary, so
;; nothing in the index references dead and it is dropped as dead.
; RUN: opt -thinlto-bc -icp-max-prom=1 %t/foo.ll -o %t/foo.o
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o

; RUN: llvm-dis %t/foo.o -o - | FileCheck %s --check-prefix=SUMMARY
; SUMMARY: gv: (name: "foo", {{.*}} calls: ((callee: ^{{[0-9]+}}, hotness: none, indirect: 1))

;; bar and dead are both declared in foo's module, where regular ICP finds them
;; in the symbol table. bar is live (it is referenced from main's module and
;; foo's summary calls it) and exported since it is declared in more than one
;; module, so the call can be promoted to its declaration. dead is not live.
; RUN: llvm-lto2 run %t/main.o %t/foo.o \
; RUN:  -icp-allow-decls -icp-max-prom=2 -icp-allow-candidate-skip \
; RUN:  -r=%t/foo.o,foo,plx \
; RUN:  -r=%t/foo.o,bar, \
; RUN:  -r=%t/foo.o,dead, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,foo, \
; RUN:  -r=%t/main.o,bar,pl \
; RUN:  -r=%t/main.o,dead,pl \
; RUN:  -r=%t/main.o,fptr,plx \
; RUN:  -thinlto-threads=1 \
; RUN:  -pass-remarks=pgo-icall-prom -pass-remarks-missed=pgo-icall-prom \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=MAIN
; RUN: llvm-nm %t.out.2 | FileCheck %s --check-prefix=FOO

;; In foo's module, the call is promoted to bar's declaration and not to
;; dead's, which the index says is not live.
; REMARKS-DAG: Promote indirect call to bar with count 10000 out of 19000
; REMARKS-DAG: Do not promote indirect call: target with md5sum 3241596039377620725 definition not available (count=9000)
;; foo is also imported into main's module. There, dead's definition is dropped
;; together with its declaration before ICP runs, so the target is not found.
; REMARKS-DAG: Cannot promote indirect call: target with md5sum 3241596039377620725 not found (count=9000)

; MAIN: T bar
; MAIN-NOT: dead
; FOO: U bar
; FOO-NOT: dead

;--- foo.ll
source_filename = "foo.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %fp) {
entry:
  call void %fp(), !prof !0
  ret void
}

;; Declarations left behind in the module, e.g. by dead code elimination of
;; direct references, which ICP finds in the symbol table.
declare void @bar()
declare void @dead()

!0 = !{!"VP", i32 0, i64 19000, i64 -2012135647395072713, i64 10000, i64 3241596039377620725, i64 9000}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = global ptr @bar

define i32 @main() {
entry:
  %fp = load ptr, ptr @fptr
  call void @foo(ptr %fp)
  ret i32 0
}

declare void @foo(ptr)

;; The noinline attribute prevents importing bar's definition.
define void @bar() #0 {
entry:
  ret void
}

;; Not referenced anywhere, so dead after the thin link.
define void @dead() #0 {
entry:
  ret void
}

attributes #0 = { noinline }
