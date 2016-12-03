; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/emit_imports.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=emitimports -thinlto-index %t.index.bc %t1.bc %t2.bc

; The imports file for this module contains the bitcode file for
; Inputs/emit_imports.ll
; RUN: cat %t1.bc.imports | count 1
; RUN: cat %t1.bc.imports | FileCheck %s --check-prefix=IMPORTS1
; IMPORTS1: emit_imports.ll.tmp2.bc

; The imports file for Input/emit_imports.ll is empty as it does not import anything.
; RUN: cat %t2.bc.imports | count 0

; RUN: rm -f %t*.thinlto.bc %t*.bc.imports
; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px

; RUN: cat %t1.bc.imports | count 1
; RUN: cat %t1.bc.imports | FileCheck %s --check-prefix=IMPORTS1

; The imports file for Input/emit_imports.ll is empty as it does not import anything.
; RUN: cat %t2.bc.imports | count 0

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
