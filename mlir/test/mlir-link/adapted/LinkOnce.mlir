// RUN: echo "module { llvm.mlir.global linkonce @X(8 : i32) {addr_space = 0 : i32} : i32 }" > %t.tmp.mlir
// RUN: mlir-link %s %t.tmp.mlir -o -

module {
  llvm.mlir.global linkonce @X(7 : i32) {addr_space = 0 : i32} : i32
}
