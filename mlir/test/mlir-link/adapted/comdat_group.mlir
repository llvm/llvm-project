// RUN: mlir-link %s -o - | FileCheck %s

// comdat disappears when linking
// XFAIL: *

// CHECK: llvm.comdat_selector @linkoncecomdat any
// CHECK: llvm.mlir.global linkonce @linkoncecomdat (2 : i32)
// CHECK: llvm.mlir.global linkonce @linkoncecomdat_unref_var (2 : i32) comdat(@__llvm_global_comdat::@linkoncecomdat)
// CHECK: llvm.func linkonce @linkoncecomdat_unref_func() comdat(@__llvm_global_comdat::@linkoncecomdat) {

module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @linkoncecomdat any
  }
  llvm.mlir.global linkonce @linkoncecomdat(2 : i32) comdat(@__llvm_global_comdat::@linkoncecomdat) {addr_space = 0 : i32} : i32
  llvm.mlir.global linkonce @linkoncecomdat_unref_var(2 : i32) comdat(@__llvm_global_comdat::@linkoncecomdat) {addr_space = 0 : i32} : i32
  llvm.func linkonce @linkoncecomdat_unref_func() comdat(@__llvm_global_comdat::@linkoncecomdat) {
    llvm.return
  }
  llvm.func @ref_linkoncecomdat() {
    %0 = llvm.mlir.addressof @linkoncecomdat : !llvm.ptr
    %1 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return
  }
}
