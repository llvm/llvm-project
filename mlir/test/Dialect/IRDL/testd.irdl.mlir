// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: irdl.dialect @testd {
irdl.dialect @testd {
  // CHECK: irdl.type @parametric {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.parameters(%[[v0]])
  // CHECK: }
  irdl.type @parametric {
    %0 = irdl.any
    irdl.parameters(%0)
  }

  // CHECK: irdl.type @attr_in_type_out {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.parameters(%[[v0]])
  // CHECK: }
  irdl.type @attr_in_type_out {
    %0 = irdl.any
    irdl.parameters(%0)
  }

  // CHECK: irdl.operation @eq {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   irdl.results(%[[v0]])
  // CHECK: }
  irdl.operation @eq {
    %0 = irdl.is i32
    irdl.results(%0)
  }

  // CHECK: irdl.operation @any {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.results(%[[v0]])
  // CHECK: }
  irdl.operation @any {
    %0 = irdl.any
    irdl.results(%0)
  }

  // CHECK: irdl.operation @dynbase {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   %[[v1:[^ ]*]] = irdl.parametric @parametric<%[[v0]]>
  // CHECK:   irdl.results(%[[v1]])
  // CHECK: }
  irdl.operation @dynbase {
    %0 = irdl.any
    %1 = irdl.parametric @parametric<%0>
    irdl.results(%1)
  }

  // CHECK: irdl.operation @dynparams {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v3:[^ ]*]] = irdl.parametric @parametric<%[[v0]]>
  // CHECK:   irdl.results(%[[v3]])
  // CHECK: }
  irdl.operation @dynparams {
    %0 = irdl.is i32
    %3 = irdl.parametric @parametric<%0>
    irdl.results(%3)
  }

  // CHECK: irdl.operation @constraint_vars {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.results(%[[v0]], %[[v0]])
  // CHECK: }
  irdl.operation @constraint_vars {
    %0 = irdl.any
    irdl.results(%0, %0)
  }
}
