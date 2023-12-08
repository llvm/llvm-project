; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_alias.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now. Ensures that the importer handles an external call
; from imported callanalias() to a function that is defined already in
; the dest module, but as an alias.
; RUN: opt -passes=function-import -summary-file %t3.thinlto.bc %t.bc -S | FileCheck %s

define i32 @main() #0 {
entry:
  call void @callanalias()
  ret i32 0
}

@analias = alias void (), ptr @globalfunc

define void @globalfunc() #0 {
entry:
  ret void
}

declare void @callanalias() #1
; CHECK-DAG: define available_externally void @callanalias()
