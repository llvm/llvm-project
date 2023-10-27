// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.comdat @__llvm_comdat
llvm.comdat @__llvm_comdat {
  // CHECK: llvm.comdat_selector @any_comdat any
  llvm.comdat_selector @any_comdat any
  // CHECK: llvm.comdat_selector @exactmatch_comdat exactmatch
  llvm.comdat_selector @exactmatch_comdat exactmatch
  // CHECK: llvm.comdat_selector @largest_comdat largest
  llvm.comdat_selector @largest_comdat largest
  // CHECK: llvm.comdat_selector @nodeduplicate_comdat nodeduplicate
  llvm.comdat_selector @nodeduplicate_comdat nodeduplicate
  // CHECK: llvm.comdat_selector @samesize_comdat samesize
  llvm.comdat_selector @samesize_comdat samesize
}
