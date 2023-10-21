// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation -split-input-file %s | FileCheck %s

memref.global "private" constant @__constant_4xf32 : memref<4xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]>

func.func @op_without_aliasing_and_allocation() -> memref<4xf32> {
  %0 = memref.get_global @__constant_4xf32 : memref<4xf32>
  return %0 : memref<4xf32>
}

// CHECK-LABEL: func @op_without_aliasing_and_allocation
//       CHECK:   [[GLOBAL:%.+]] = memref.get_global @__constant_4xf32
//       CHECK:   cf.assert %false{{[0-9_]*}}, "Must have ownership of operand #0"
//       CHECK:   return [[GLOBAL]] :
