// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Types that have cyclic references.

// CHECK: irdl.dialect @testd {
irdl.dialect @testd {
  // CHECK:   irdl.type @self_referencing {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   %[[v1:[^ ]*]] = irdl.parametric @testd::@self_referencing<%[[v0]]>
  // CHECK:   %[[v2:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v3:[^ ]*]] = irdl.any_of(%[[v1]], %[[v2]])
  // CHECK:   irdl.parameters(foo: %[[v3]])
  // CHECK: }
  irdl.type @self_referencing {
    %0 = irdl.any
    %1 = irdl.parametric @testd::@self_referencing<%0>
    %2 = irdl.is i32
    %3 = irdl.any_of(%1, %2)
    irdl.parameters(foo: %3)
  }


  // CHECK:   irdl.type @type1 {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   %[[v1:[^ ]*]] = irdl.parametric @testd::@type2<%[[v0]]>
  // CHECK:   %[[v2:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v3:[^ ]*]] = irdl.any_of(%[[v1]], %[[v2]])
  // CHECK:   irdl.parameters(foo: %[[v3]])
  irdl.type @type1 {
    %0 = irdl.any
    %1 = irdl.parametric @testd::@type2<%0>
    %2 = irdl.is i32
    %3 = irdl.any_of(%1, %2)
    irdl.parameters(foo: %3)
  }

  // CHECK:   irdl.type @type2 {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   %[[v1:[^ ]*]] = irdl.parametric @testd::@type1<%[[v0]]>
  // CHECK:   %[[v2:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v3:[^ ]*]] = irdl.any_of(%[[v1]], %[[v2]])
  // CHECK:   irdl.parameters(foo: %[[v3]])
  irdl.type @type2 {
      %0 = irdl.any
      %1 = irdl.parametric @testd::@type1<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(foo: %3)
  }
}
