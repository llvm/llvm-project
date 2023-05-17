// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  // CHECK-LABEL: irdl.dialect @cmath {
  irdl.dialect @cmath {

    // CHECK: irdl.type @complex {
    // CHECK:   %[[v0:[^ ]*]] = irdl.is f32
    // CHECK:   irdl.parameters(%[[v0]])
    // CHECK: }
    irdl.type @complex {
      %0 = irdl.is f32
      irdl.parameters(%0)
    }

    // CHECK: irdl.operation @norm {
    // CHECK:   %[[v0:[^ ]*]] = irdl.any
    // CHECK:   %[[v1:[^ ]*]] = irdl.parametric @complex<%[[v0]]>
    // CHECK:   irdl.operands(%[[v1]])
    // CHECK:   irdl.results(%[[v0]])
    // CHECK: }
    irdl.operation @norm {
      %0 = irdl.any
      %1 = irdl.parametric @complex<%0>
      irdl.operands(%1)
      irdl.results(%0)
    }

    // CHECK: irdl.operation @mul {
    // CHECK:   %[[v0:[^ ]*]] = irdl.is f32
    // CHECK:   %[[v3:[^ ]*]] = irdl.parametric @complex<%[[v0]]>
    // CHECK:   irdl.operands(%[[v3]], %[[v3]])
    // CHECK:   irdl.results(%[[v3]])
    // CHECK: }
    irdl.operation @mul {
      %0 = irdl.is f32
      %3 = irdl.parametric @complex<%0>
      irdl.operands(%3, %3)
      irdl.results(%3)
    }

  }
}
