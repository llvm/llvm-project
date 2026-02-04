// RUN: mlir-opt %s --sparsification-and-bufferization | FileCheck %s

module {
  func.func @main() {
    %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerAttr>($_self)"
    %1 = sparse_tensor.has_runtime_library
    %2 = math.ctlz %1 : i1
    %3 = math.ipowi %2, %1 : i1
    %4 = emitc.unary_minus %3 : (i1) -> i1
    func.return
  }
}

// CHECK-LABEL: func.func @main
// CHECK: emitc.unary_minus
