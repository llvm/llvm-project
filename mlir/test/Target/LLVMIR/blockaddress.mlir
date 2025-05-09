// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file | FileCheck %s

llvm.mlir.global private @g() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 0>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @fn() {
  llvm.br ^bb1
^bb1:
  llvm.blocktag <id = 0>
  llvm.return
}

// CHECK: @g = private global ptr blockaddress(@fn, %1)
// CHECK: define void @fn() {
// CHECK:   br label %[[RET:.*]]
// CHECK: [[RET]]:
// CHECK:   ret void
// CHECK: }

// -----

llvm.func @blockaddr0() -> !llvm.ptr {
  %0 = llvm.blockaddress <function = @blockaddr0, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  llvm.blocktag <id = 1>
  llvm.return %0 : !llvm.ptr
}

// CHECK: define ptr @blockaddr0() {
// CHECK:   br label %[[RET:.*]]
// CHECK: [[RET]]:
// CHECK:   ret ptr blockaddress(@blockaddr0, %1)
// CHECK: }
