// RUN: mlir-opt %s -pass-pipeline='builtin.module(llvm.func(canonicalize{region-simplify=aggressive}))' -verify-diagnostics -split-input-file | FileCheck %s

llvm.mlir.global private @x() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
  %0 = llvm.blockaddress <function = @ba, tag = <id = 2>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @ba()
llvm.func @ba() -> !llvm.ptr {
  %0 = llvm.blockaddress <function = @ba, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  // CHECK: llvm.blocktag <id = 1>
  llvm.blocktag <id = 1>
  llvm.br ^bb2
^bb2:
  // CHECK: llvm.blocktag <id = 2>
  llvm.blocktag <id = 2>
  llvm.return %0 : !llvm.ptr
}

// -----


llvm.mlir.global private @g() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 0>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.global private @h() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 1>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @fn
llvm.func @fn(%cond : i1, %arg0 : i32, %arg1 : i32) -> i32 {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: llvm.blocktag <id = 0>
  // CHECK: llvm.return
  llvm.blocktag <id = 0>
  llvm.return %arg0 : i32
^bb2:
  // CHECK: llvm.blocktag <id = 1>
  // CHECK: llvm.return
  llvm.blocktag <id = 1>
  llvm.return %arg1 : i32
}

// -----

llvm.mlir.global private @g() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 1>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// The Canonicalizer's region simplify pass can be hazardous when dealing
// with indirect branches, as there is currently no mechanism to convey
// dialect-specific block constraints.

llvm.func @fn(%dest : !llvm.ptr, %arg0 : i32, %arg1 : i32) -> i32 {
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head
  ]
^head:
  llvm.blocktag <id = 0>
  llvm.return %arg0 : i32
^tail:
  // expected-error@+1 {{not allowed in unrecheable blocks}}
  llvm.blocktag <id = 1>
  llvm.return %arg1 : i32
}
