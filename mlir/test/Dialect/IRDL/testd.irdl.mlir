// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: irdl.dialect @testd {
irdl.dialect @testd {
  // CHECK: irdl.type @parametric {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.parameters(foo: %[[v0]])
  // CHECK: }
  irdl.type @parametric {
    %0 = irdl.any
    irdl.parameters(foo: %0)
  }

  // CHECK: irdl.attribute @parametric_attr {
  // CHECK:  %[[v0:[^ ]*]] = irdl.any
  // CHECK:  irdl.parameters(foo: %[[v0]])
  // CHECK: }
  irdl.attribute @parametric_attr {
    %0 = irdl.any
    irdl.parameters(foo: %0)
  }

  // CHECK: irdl.type @attr_in_type_out {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.parameters(foo: %[[v0]])
  // CHECK: }
  irdl.type @attr_in_type_out {
    %0 = irdl.any
    irdl.parameters(foo: %0)
  }

  // CHECK: irdl.operation @eq {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   irdl.results(foo: %[[v0]])
  // CHECK: }
  irdl.operation @eq {
    %0 = irdl.is i32
    irdl.results(foo: %0)
  }

  // CHECK: irdl.operation @anyof {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results(foo: %[[v2]])
  // CHECK: }
  irdl.operation @anyof {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.results(foo: %2)
  }

  // CHECK: irdl.operation @all_of {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.all_of(%[[v2]], %[[v1]])
  // CHECK:   irdl.results(foo: %[[v3]])
  // CHECK: }
  irdl.operation @all_of {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.all_of(%2, %1)
    irdl.results(foo: %3)
  }

  // CHECK: irdl.operation @any {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   irdl.results(foo: %[[v0]])
  // CHECK: }
  irdl.operation @any {
    %0 = irdl.any
    irdl.results(foo: %0)
  }

  // CHECK: irdl.operation @dyn_type_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base @testd::@parametric
  // CHECK:   irdl.results(foo: %[[v1]])
  // CHECK: }
  irdl.operation @dyn_type_base {
    %0 = irdl.base @testd::@parametric
    irdl.results(foo: %0)
  }

  // CHECK: irdl.operation @dyn_attr_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base @testd::@parametric_attr
  // CHECK:   irdl.attributes {"attr1" = %[[v1]]}
  // CHECK: }
  irdl.operation @dyn_attr_base {
    %0 = irdl.base @testd::@parametric_attr
    irdl.attributes {"attr1" = %0}
  }

  // CHECK: irdl.operation @named_type_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base "!builtin.integer"
  // CHECK:   irdl.results(foo: %[[v1]])
  // CHECK: }
  irdl.operation @named_type_base {
    %0 = irdl.base "!builtin.integer"
    irdl.results(foo: %0)
  }

  // CHECK: irdl.operation @named_attr_base {
  // CHECK:   %[[v1:[^ ]*]] = irdl.base "#builtin.integer"
  // CHECK:   irdl.attributes {"attr1" = %[[v1]]}
  // CHECK: }
  irdl.operation @named_attr_base {
    %0 = irdl.base "#builtin.integer"
    irdl.attributes {"attr1" = %0}
  }

  // CHECK: irdl.operation @dynparams {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   %[[v3:[^ ]*]] = irdl.parametric @testd::@parametric<%[[v2]]>
  // CHECK:   irdl.results(foo: %[[v3]])
  // CHECK: }
  irdl.operation @dynparams {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    %3 = irdl.parametric @testd::@parametric<%2>
    irdl.results(foo: %3)
  }

  // CHECK: irdl.operation @constraint_vars {
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[v2:[^ ]*]] = irdl.any_of(%[[v0]], %[[v1]])
  // CHECK:   irdl.results(foo: %[[v2]], bar: %[[v2]])
  // CHECK: }
  irdl.operation @constraint_vars {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.results(foo: %2, bar: %2)
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
  // CHECK: irdl.operation @regions {
  // CHECK:   %[[r0:[^ ]*]] = irdl.region
  // CHECK:   %[[v0:[^ ]*]] = irdl.is i32
  // CHECK:   %[[v1:[^ ]*]] = irdl.is i64
  // CHECK:   %[[r1:[^ ]*]] = irdl.region(%[[v0]], %[[v1]])
  // CHECK:   %[[r2:[^ ]*]] = irdl.region with size 3
  // CHECK:   %[[r3:[^ ]*]] = irdl.region()
  // CHECK:   irdl.regions(foo: %[[r0]], bar: %[[r1]], baz: %[[r2]], qux: %[[r3]])
  // CHECK: }
  irdl.operation @regions {
    %r0 = irdl.region
    %v0 = irdl.is i32
    %v1 = irdl.is i64
    %r1 = irdl.region(%v0, %v1)
    %r2 = irdl.region with size 3
    %r3 = irdl.region()

    irdl.regions(foo: %r0, bar: %r1, baz: %r2, qux: %r3)
  }

  // CHECK: irdl.operation @region_and_operand {
  // CHECK:   %[[v0:[^ ]*]] = irdl.any
  // CHECK:   %[[r0:[^ ]*]] = irdl.region(%[[v0]])
  // CHECK:   irdl.operands(foo: %[[v0]])
  // CHECK:   irdl.regions(bar: %[[r0]])
  // CHECK: }
  irdl.operation @region_and_operand {
    %v0 = irdl.any
    %r0 = irdl.region(%v0)

    irdl.operands(foo: %v0)
    irdl.regions(bar: %r0)
  }
}
