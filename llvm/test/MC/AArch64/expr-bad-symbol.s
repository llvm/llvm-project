// RUN: not llvm-mc -triple aarch64-- %s 2>&1 | FileCheck %s

  .text
_foo:
  str q28, [x0, #1*6*4*@]
// CHECK: error: expected a symbol reference
