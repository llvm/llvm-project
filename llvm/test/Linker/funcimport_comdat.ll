; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_comdat.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Ensure linking of comdat containing external linkage global and function
; removes the imported available_externally defs from comdat.
; RUN: llvm-link %t2.bc -summary-index=%t3.thinlto.bc -import=comdat1_func1:%t.bc -S | FileCheck %s --check-prefix=IMPORTCOMDAT
; IMPORTCOMDAT-NOT: $comdat1 = comdat any
; IMPORTCOMDAT-NOT: comdat($comdat1)

; Ensure linking of comdat containing internal linkage function with alias
; removes the imported and promoted available_externally defs from comdat.
; RUN: llvm-link %t2.bc -summary-index=%t3.thinlto.bc -import=comdat2_func1:%t.bc -S | FileCheck %s --check-prefix=IMPORTCOMDAT2
; IMPORTCOMDAT2-NOT: $comdat2 = comdat any
; IMPORTCOMDAT2-NOT: comdat($comdat2)

$comdat1 = comdat any
@comdat1_glob = global i32 0, comdat($comdat1)
define void @comdat1_func1() comdat($comdat1) {
  ret void
}

$comdat2 = comdat any
@comdat2_alias = alias void (), void ()* @comdat2_func1
define internal void @comdat2_func1() comdat($comdat2) {
  ret void
}
