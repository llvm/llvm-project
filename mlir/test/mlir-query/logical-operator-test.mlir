// RUN: mlir-query %s -c "m allOf(hasOpName(\"memref.alloca\"), hasOpAttrName(\"alignment\"))" | FileCheck %s

func.func @dynamic_alloca(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %0 = memref.alloca(%arg0, %arg1) : memref<?x?xf32>
  memref.alloca(%arg0, %arg1) {alignment = 32} : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// CHECK: Match #1:
// CHECK: {{.*}}.mlir:5:3: note: "root" binds here
// CHECK: memref.alloca(%arg0, %arg1) {alignment = 32} : memref<?x?xf32>
