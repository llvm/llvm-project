// RUN: mlir-opt -llvm-add-comdats -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.comdat @__llvm_comdat {
// CHECK-DAG: llvm.comdat_selector @linkonce any
// CHECK-DAG: llvm.comdat_selector @linkonce_odr any
// CHECK: }

// CHECK: llvm.func linkonce @linkonce() comdat(@__llvm_comdat::@linkonce)
llvm.func linkonce @linkonce() {
  llvm.return
}

// CHECK: llvm.func linkonce_odr @linkonce_odr() comdat(@__llvm_comdat::@linkonce_odr)
llvm.func linkonce_odr @linkonce_odr() {
  llvm.return
}

