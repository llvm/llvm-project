// RUN: mlir-opt %s -vector-bufferize -split-input-file -verify-diagnostics
// | FileCheck %s

// CHECK-LABEL: func @mask(
func.func @mask(%t0: tensor<?xf32>, %val: vector<16xf32>, %idx: index, %m0: vector<16xi1>) -> tensor<?xf32> {
  // expected-error @+1 {{'vector.mask' op body must bufferize in-place}}
  %0 = vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
