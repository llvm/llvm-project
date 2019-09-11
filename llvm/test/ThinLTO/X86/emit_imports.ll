; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/emit_imports.ll -o %t2.bc
; RUN: opt -module-summary %p/Inputs/emit_imports2.ll -o %t2b.bc
; Include a file with an empty module summary index, to ensure that the expected
; output files are created regardless, for a distributed build system.
; RUN: opt -module-summary %p/Inputs/empty.ll -o %t3.bc
; RUN: rm -f %t3.bc.imports
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc %t2b.bc %t3.bc
; RUN: llvm-lto -thinlto-action=emitimports -thinlto-index %t.index.bc %t1.bc %t2.bc %t2b.bc %t3.bc

; The imports file for this module contains the bitcode file for
; Inputs/emit_imports.ll
; RUN: cat %t1.bc.imports | count 2
; RUN: cat %t1.bc.imports | FileCheck %s --check-prefix=IMPORTS1
; IMPORTS1: emit_imports.ll.tmp2.bc
; IMPORTS1: emit_imports.ll.tmp2b.bc

; The imports file for Input/emit_imports.ll is empty as it does not import anything.
; RUN: cat %t2.bc.imports | count 0

; The imports file for Input/empty.ll is empty but should exist.
; RUN: cat %t3.bc.imports | count 0

; RUN: rm -f %t1.thinlto.bc %t1.bc.imports
; RUN: rm -f %t2.thinlto.bc %t2.bc.imports
; RUN: rm -f %t3.bc.thinlto.bc %t3.bc.imports
; RUN: llvm-lto2 run %t1.bc %t2.bc %t2b.bc %t3.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,h, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px \
; RUN:     -r=%t2b.bc,h,px

; RUN: cat %t1.bc.imports | count 2
; RUN: cat %t1.bc.imports | FileCheck %s --check-prefix=IMPORTS1

; The imports file for Input/emit_imports.ll is empty as it does not import anything.
; RUN: cat %t2.bc.imports | count 0

; The imports file for Input/empty.ll is empty but should exist.
; RUN: cat %t3.bc.imports | count 0

; The index file should be created even for the input with an empty summary.
; RUN: ls %t3.bc.thinlto.bc

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)
declare void @h(...)

define void @f() {
entry:
  call void (...) @g()
  call void (...) @h()
  ret void
}
