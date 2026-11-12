// RUN: mlir-opt --canonicalize %s | FileCheck %s


module {

// CHECK:       func.func @f() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
  func.func @f() {
    return
  ^bb1:  // no predecessors
    omp.parallel   {
      %0 = llvm.intr.stacksave : !llvm.ptr
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      omp.terminator
    }
    return
  }

}
