// RUN: toyc-ch7 %s -emit=mlir -opt 2>&1 | FileCheck %s

toy.func @main() {
  %0 = toy.struct_constant [
    [dense<4.000000e+00> : tensor<2x2xf64>], dense<4.000000e+00> : tensor<2x2xf64>
  ] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>>
  %1 = toy.struct_access %0[0] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>> -> !toy.struct<tensor<*xf64>>
  %2 = toy.struct_access %1[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
  toy.print %2 : tensor<*xf64>
  toy.return
}

// CHECK-LABEL: toy.func @main
// CHECK-NEXT: %[[CST:.*]] = toy.constant dense<4.0
// CHECK-NEXT: toy.print %[[CST]]
