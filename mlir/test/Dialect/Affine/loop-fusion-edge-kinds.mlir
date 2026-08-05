// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-fusion=test-loop-fusion-edge-kinds | FileCheck %s

// CHECK: edge-kinds ssa-only any=1 memory=0 ssa=1 out=1 in=1 count=1
// CHECK: edge-kinds both any=1 memory=1 ssa=1 out=2 in=2 count=2
// CHECK: edge-kinds ssa-after-memory-removal any=1 memory=0 ssa=1 out=1 in=1 count=1
// CHECK: edge-kinds empty any=0 memory=0 ssa=0 out=0 in=0 count=0
func.func @edge_kinds() {
  %src = memref.alloc() : memref<1xi64>
  %dst = memref.alloc() : memref<1xi64>
  return
}
