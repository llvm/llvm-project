; Test to ensure that building summary with -force-summary-edges-cold
; blocks importing as expected.

; "-stats" and "-debug-only" require +Asserts.
; REQUIRES: asserts

; First do with default options, which should import
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_forcecold.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc
; RUN: opt -passes=function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=IMPORT

; Next rebuild caller module summary with non-critical edges forced cold (which
; should affect all edges in this test as we don't have any sample pgo).
; Make sure we don't import.
; RUN: opt -force-summary-edges-cold=all-non-critical -module-summary %s -o %t.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc
; RUN: opt -passes=function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=NOIMPORT

; Next rebuild caller module summary with all edges forced cold.
; Make sure we don't import.
; RUN: opt -force-summary-edges-cold=all -module-summary %s -o %t.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc
; RUN: opt -passes=function-import -stats -print-imports -summary-file %t3.thinlto.bc %t.bc -S 2>&1 | FileCheck %s --check-prefix=NOIMPORT

define i32 @main() {
entry:
  call void @foo()
  ret i32 0
}

; IMPORT: Import foo
; NOIMPORT-NOT: Import foo
; IMPORT: define available_externally void @foo()
; NOIMPORT: declare void @foo()
declare void @foo()
