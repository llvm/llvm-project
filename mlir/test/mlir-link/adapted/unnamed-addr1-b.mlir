// RUN: true

module {
  llvm.mlir.global common unnamed_addr @"global-c"(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @"global-d"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @"global-e"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @"global-f"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global common @"global-g"(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @"global-h"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @"global-i"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @"global-j"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.alias external unnamed_addr @"alias-a" : i32 {
    %0 = llvm.mlir.addressof @"global-f" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.alias external unnamed_addr @"alias-b" : i32 {
    %0 = llvm.mlir.addressof @"global-f" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.alias external @"alias-c" : i32 {
    %0 = llvm.mlir.addressof @"global-f" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.alias external @"alias-d" : i32 {
    %0 = llvm.mlir.addressof @"global-f" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func weak unnamed_addr @"func-c"() {
    llvm.return
  }
  llvm.func weak unnamed_addr @"func-d"() {
    llvm.return
  }
  llvm.func weak unnamed_addr @"func-e"() {
    llvm.return
  }
  llvm.func weak @"func-g"() {
    llvm.return
  }
  llvm.func weak @"func-h"() {
    llvm.return
  }
  llvm.func weak @"func-i"() {
    llvm.return
  }
}
