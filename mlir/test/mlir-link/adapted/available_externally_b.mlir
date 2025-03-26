// RUN: true
module {
  llvm.mlir.global external hidden unnamed_addr constant @foo(0 : i32) {addr_space = 0 : i32, dso_local} : i32
}

