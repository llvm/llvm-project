// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  // CHECK-LABEL: irdl.dialect @testd {
  irdl.dialect @testd {
    // CHECK: irdl.type @singleton
    irdl.type @singleton

    // CHECK: irdl.type @parametrized {
    // CHECK:   %[[v0:[^ ]*]] = irdl.any
    // CHECK:   %[[v1:[^ ]*]] = irdl.is i32
    // CHECK:   %[[v2:[^ ]*]] = irdl.is i64
    // CHECK:   %[[v3:[^ ]*]] = irdl.any_of(%[[v1]], %[[v2]])
    // CHECK:   irdl.parameters(%[[v0]], %[[v3]])
    // CHECK: }
    irdl.type @parametrized {
      %0 = irdl.any
      %1 = irdl.is i32
      %2 = irdl.is i64
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(%0, %3)
    }

    // CHECK: irdl.operation @any {
    // CHECK:   %[[v0:[^ ]*]] = irdl.any
    // CHECK:   irdl.results(%[[v0]])
    // CHECK: }
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(%0)
    }
  }
}
