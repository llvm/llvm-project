// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  // CHECK-LABEL: irdl.dialect @cmath {
  irdl.dialect @cmath {

    // CHECK: irdl.type @complex {
    // CHECK:   %[[v0:[^ ]*]] = irdl.is f32
    // CHECK:   %[[v1:[^ ]*]] = irdl.is f64
    // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
    // CHECK:   irdl.parameters(elem: %[[v2]])
    // CHECK: }
    irdl.type @complex {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      irdl.parameters(elem: %2)
    }

    // CHECK: irdl.operation @norm {
    // CHECK:   %[[v0:[^ ]*]] = irdl.any
    // CHECK:   %[[v1:[^ ]*]] = irdl.parametric @cmath::@complex<%[[v0]]>
    // CHECK:   irdl.operands(complex: %[[v1]])
    // CHECK:   irdl.results(norm: %[[v0]])
    // CHECK: }
    irdl.operation @norm {
      %0 = irdl.any
      %1 = irdl.parametric @cmath::@complex<%0>
      irdl.operands(complex: %1)
      irdl.results(norm: %0)
    }

    // CHECK: irdl.operation @mul {
    // CHECK:   %[[v0:[^ ]*]] = irdl.is f32
    // CHECK:   %[[v1:[^ ]*]] = irdl.is f64
    // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
    // CHECK:   %[[v3:[^ ]*]] = irdl.parametric @cmath::@complex<%[[v2]]>
    // CHECK:   irdl.operands(lhs: %[[v3]], rhs: %[[v3]])
    // CHECK:   irdl.results(res: %[[v3]])
    // CHECK: }
    irdl.operation @mul {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @cmath::@complex<%2>
      irdl.operands(lhs: %3, rhs: %3)
      irdl.results(res: %3)
    }

  }
}
