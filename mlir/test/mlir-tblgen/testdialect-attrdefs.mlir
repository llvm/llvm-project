// RUN: mlir-opt %s | mlir-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func private @compoundA()
// CHECK-SAME: #test.cmpnd_a<1, !test.smpla, [5, 6]>
func private @compoundA() attributes {foo = #test.cmpnd_a<1, !test.smpla, [5, 6]>}
