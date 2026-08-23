// RUN: tr-opt --show-dialects %s | FileCheck %s --check-prefix=DIALECT
// RUN: tr-opt %s | FileCheck %s

// Milestone 1: the driver registers `tr` (plus func/arith from the canonical
// source) and can parse a module. Types and ops are Milestone 2 / 3.

// DIALECT: Available Dialects: affine,arith,builtin,cf,func,gpu,index,linalg,llvm,memref,nvvm,scf,tr,transform

module {
  // CHECK-LABEL: func.func @smoke
  func.func @smoke() {
    %c128 = arith.constant 128 : index
    // CHECK: arith.constant 128 : index
    return
  }
}
