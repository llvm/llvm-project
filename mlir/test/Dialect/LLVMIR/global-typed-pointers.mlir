// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.mlir.global internal @global(42 : i64) {addr_space = 0 : i32} : i64
llvm.mlir.global internal @global(42 : i64) : i64

// CHECK: llvm.mlir.global internal constant @".string"("foobar")
llvm.mlir.global internal constant @".string"("foobar") : !llvm.array<6 x i8>

func.func @references() {
  // CHECK: llvm.mlir.addressof @global : !llvm.ptr<i64>
  %0 = llvm.mlir.addressof @global : !llvm.ptr<i64>

  // CHECK: llvm.mlir.addressof @".string" : !llvm.ptr<array<6 x i8>>
  %1 = llvm.mlir.addressof @".string" : !llvm.ptr<array<6 x i8>>

  llvm.return
}

// -----

llvm.mlir.global internal @foo(0: i32) : i32

func.func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referenced global}}
  llvm.mlir.addressof @foo : !llvm.ptr<i64>
  llvm.return
}

// -----

llvm.func @foo()

llvm.func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referenced function}}
  llvm.mlir.addressof @foo : !llvm.ptr<i8>
  llvm.return
}

// -----

llvm.mlir.global internal @g(32 : i64) {addr_space = 3: i32} : i64
func.func @mismatch_addr_space() {
  // expected-error @+1 {{pointer address space must match address space of the referenced global}}
  llvm.mlir.addressof @g : !llvm.ptr<i64, 4>
  llvm.return
}
