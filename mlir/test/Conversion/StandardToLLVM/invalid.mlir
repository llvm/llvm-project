// RUN: mlir-opt %s -verify-diagnostics -split-input-file

#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func @invalid_memref_cast(%arg0: memref<?x?xf64>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  // expected-error@+1: 'std.memref_cast' op operand #0 must be unranked.memref of any type values or memref of any type values,
  %5 = memref_cast %arg0 : memref<?x?xf64> to memref<?x?xf64, #map1>
  %25 = std.subview %5[%c0, %c0][%c1, %c1][] : memref<?x?xf64, #map1> to memref<?x?xf64, #map1>
  return
}

