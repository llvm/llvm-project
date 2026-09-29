; Test to ensure that thin linking with -import-cutoff stops importing when
; expected.

; "-stats" and "-debug-only" require +Asserts.
; REQUIRES: asserts

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_cutoff.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; First do with default options, which should import both foo and bar
; RUN: opt -passes=function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=IMPORT

; Next try to restrict to 1 import. This should import just foo.
; RUN: opt -import-cutoff=1 -passes=function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=IMPORT1

; Next try to restrict to 0 imports. This should not import.
; RUN: opt -import-cutoff=0 -passes=function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=NOIMPORT

define i32 @main() {
entry:
  call void @foo()
  call void @bar()
  ret i32 0
}

declare void @foo()
declare void @bar()

; Check -print-imports output
; IMPORT: Import foo
; IMPORT: Import bar
; IMPORT1: Import foo
; IMPORT1-NOT: Import bar
; NOIMPORT-NOT: Import foo
; NOIMPORT-NOT: Import bar

; Check -S output
; IMPORT-DAG: define available_externally void @foo()
; IMPORT-DAG: define available_externally void @bar()
; NOIMPORT-DAG: declare void @foo()
; NOIMPORT-DAG: declare void @bar()
; IMPORT1-DAG: define available_externally void @foo()
; IMPORT1-DAG: declare void @bar()

; Check -stats output
; IMPORT: 2 function-import - Number of functions imported
; IMPORT1: 1 function-import - Number of functions imported
; NOIMPORT-NOT: function-import - Number of functions imported
