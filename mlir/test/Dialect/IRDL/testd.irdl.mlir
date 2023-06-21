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

  // CHECK: irdl.operation @anyof {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results(%[[v2]])
  // CHECK: }
  irdl.operation @anyof {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.results(%2)
  }

  // CHECK: irdl.operation @all_of {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.all_of(%[[v2]], %[[v1]])
  // CHECK:   irdl.results(%[[v3]])
  // CHECK: }
  irdl.operation @all_of {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.all_of(%2, %1)
    irdl.results(%3)
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
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.parametric @parametric<%[[v2]]>
  // CHECK:   irdl.results(%[[v3]])
  // CHECK: }
  irdl.operation @dynparams {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.parametric @parametric<%2>
    irdl.results(%3)
  }

  // CHECK: irdl.operation @constraint_vars {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results(%[[v2]], %[[v2]])
  // CHECK: }
  irdl.operation @constraint_vars {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.results(%2, %2)
  }

  // CHECK: irdl.operation @attrs {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   irdl.attributes {"attr1" = %[[v0]], "attr2" = %[[v1]]}
  // CHECK: }
  irdl.operation @attrs {
    %0 = irdl.is i32
    %1 = irdl.is i64

    irdl.attributes {
      "attr1" = %0,
      "attr2" = %1
    }
  }
}
