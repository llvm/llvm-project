// RUN: mlir-opt -emit-bytecode %s | mlir-opt --mlir-print-debuginfo | FileCheck %s

// Verify that the distinct attribute which is used transitively
// through two aliases does not end up duplicated when round-tripped
// through bytecode.

// CHECK: distinct[0]
// CHECK-NOT: distinct[1]
#attr_ugly = #test<attr_ugly begin distinct[0]<> end>
#attr_ugly1 = #test<attr_ugly begin #attr_ugly end>

module attributes {test.alias = #attr_ugly, test.alias1 = #attr_ugly1} {
}