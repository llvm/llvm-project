// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: llvm.func @ba()
llvm.func @ba() -> !llvm.ptr {
  %0 = llvm.blockaddress <function = @ba, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  // CHECK: llvm.blocktag <id = 1>
  llvm.blocktag <id = 1>
  llvm.return %0 : !llvm.ptr
}
