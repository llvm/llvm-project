// RUN: mlir-link %s -split-input-file -o - | FileCheck %s

module {
  // CHECK-DAG: llvm.mlir.global common @"global-a"(0 : i32)
  llvm.mlir.global common @"global-a"(0 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global common unnamed_addr @"global-b"(0 : i32)
  llvm.mlir.global common unnamed_addr @"global-b"(0 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global common unnamed_addr @"global-c"(0 : i32)
  llvm.mlir.global common unnamed_addr @"global-c"(0 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external @"global-d"(42 : i32)
  llvm.mlir.global external @"global-d"() {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external unnamed_addr @"global-e"(42 : i32)
  llvm.mlir.global external unnamed_addr @"global-e"() {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external @"global-f"(42 : i32)
  llvm.mlir.global weak @"global-f"(42 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global common @"global-g"(0 : i32)
  llvm.mlir.global common unnamed_addr @"global-g"(0 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external @"global-h"(42 : i32)
  llvm.mlir.global external @"global-h"() {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external @"global-i"(42 : i32)
  llvm.mlir.global external unnamed_addr @"global-i"() {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.mlir.global external @"global-j"(42 : i32)
  llvm.mlir.global weak @"global-j"(42 : i32) {addr_space = 0 : i32} : i32

  // CHECK-DAG: llvm.func weak @"func-a"() {
  llvm.func weak @"func-a"() {
    llvm.return
  }

  // CHECK-DAG: llvm.func weak unnamed_addr @"func-b"() {
  llvm.func weak unnamed_addr @"func-b"() {
    llvm.return
  }

  llvm.func @"use-global-d"() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @"global-d" : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  // CHECK-DAG: llvm.func weak @"func-c"() {
  llvm.func @"func-c"()

  llvm.func @"use-func-c"() {
    llvm.call @"func-c"() : () -> ()
    llvm.return
  }

  // CHECK-DAG: llvm.func weak @"func-d"() {
  llvm.func weak @"func-d"() {
    llvm.return
  }

  // CHECK-DAG llvm.func weak unnamed_addr @"func-e"() {
  llvm.func weak unnamed_addr @"func-e"() {
    llvm.return
  }

  // CHECK-DAG: llvm.func weak @"func-g"() {
  llvm.func @"func-g"()

  // CHECK-DAG: llvm.func weak @"func-h"() {
  llvm.func weak @"func-h"() {
    llvm.return
  }

  // CHECK-DAG: llvm.func weak @"func-i"() {
  llvm.func weak unnamed_addr @"func-i"() {
    llvm.return
  }
}

// -----

module {
  llvm.mlir.global common unnamed_addr @"global-c"(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @"global-d"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @"global-e"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external unnamed_addr @"global-f"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global common @"global-g"(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @"global-h"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @"global-i"(42 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @"global-j"(42 : i32) {addr_space = 0 : i32} : i32
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
