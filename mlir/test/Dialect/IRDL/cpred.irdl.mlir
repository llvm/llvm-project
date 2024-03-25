// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  // CHECK-LABEL: irdl.dialect @dialect {
  irdl.dialect @dialect {
    // CHECK-LABEL: irdl.type @type {
    irdl.type @type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerAttr>($_self)"
      // CHECK: %{{.*}} = irdl.c_pred "::llvm::isa<::mlir::IntegerAttr>($_self)"
      irdl.parameters(%0)
      // CHECK: irdl.parameters(%{{.*}})
    }
  }
}
