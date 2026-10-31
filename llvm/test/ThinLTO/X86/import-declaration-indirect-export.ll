;; Test that with -import-declaration, a function imported as a declaration for
;; an indirect call edge is exported from its defining module, so that it is
;; not internalized. Otherwise, if the backend promoted the indirect call to the
;; imported declaration (here via regular ICP under -icp-allow-decls), there
;; would be an undefined reference to the internalized function.

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -o %t/foo.o

;; The indirect call edge from foo to bar is marked indirect in the summary.
; RUN: llvm-dis %t/foo.o -o - | FileCheck %s --check-prefix=SUMMARY
; SUMMARY: gv: (name: "foo", {{.*}} calls: ((callee: ^{{[0-9]+}}, hotness: none, indirect: 1))

;; Note that bar is not visible outside the LTO unit (no x in its resolution),
;; and it is only referenced in main's module via its address (in the
;; initializer of @fptr), so it would be internalized if it wasn't exported.
; RUN: llvm-lto2 run %t/main.o %t/foo.o \
; RUN:  -import-declaration -icp-allow-decls \
; RUN:  -r=%t/foo.o,foo,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,foo, \
; RUN:  -r=%t/main.o,bar,pl \
; RUN:  -r=%t/main.o,fptr,plx \
; RUN:	-thinlto-threads=1 \
; RUN:  -pass-remarks=pgo-icall-prom \
; RUN:  -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS
; RUN: llvm-dis %t.out.2.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=EXPORTED
; RUN: llvm-nm %t.out.2 | FileCheck %s --check-prefix=REFERENCED

;; Regular ICP promoted the indirect call in foo to the imported declaration of
;; bar.
; REMARKS: Promote indirect call to bar with count 10000 out of 10000

; IMPORT: declare !guid !{{[0-9]+}} void @bar()

;; bar is exported and therefore not internalized.
; EXPORTED: T bar
;; foo's module references bar after the promotion.
; REFERENCED: U bar

;; Without -import-declaration, bar is not imported into foo's module, so the
;; indirect call there can't be promoted, and bar is internalized.
; RUN: llvm-lto2 run %t/main.o %t/foo.o \
; RUN:  -icp-allow-decls \
; RUN:  -r=%t/foo.o,foo,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,foo, \
; RUN:  -r=%t/main.o,bar,pl \
; RUN:  -r=%t/main.o,fptr,plx \
; RUN:	-thinlto-threads=1 \
; RUN:  -pass-remarks-missed=pgo-icall-prom \
; RUN:  -o %t.nodecl.out 2>&1 | FileCheck %s --check-prefix=REMARKS-NODECL
; RUN: llvm-nm %t.nodecl.out.1 | FileCheck %s --check-prefix=INTERNALIZED

; REMARKS-NODECL: Cannot promote indirect call: target with md5sum 16434608426314478903 not found (count=10000)
; INTERNALIZED: t bar

;--- foo.ll
source_filename = "foo.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %fp) {
entry:
  call void %fp(), !prof !0
  ret void
}

!0 = !{!"VP", i32 0, i64 10000, i64 -2012135647395072713, i64 10000}

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

attributes #0 = { noinline }
