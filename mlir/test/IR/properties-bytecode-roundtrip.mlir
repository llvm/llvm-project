// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// CHECK-LABEL: "test.using_int_property_with_worse_bytecode"
// CHECK-SAME: value = 3
"test.using_int_property_with_worse_bytecode"() <{value = 3}> : () -> ()
